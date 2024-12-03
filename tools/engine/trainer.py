import copy
import datetime
import os
import random
import time

import numpy as np
import torch
from tqdm import tqdm

from openrec.losses import build_loss
from openrec.metrics import build_metric
from openrec.modeling import build_model
from openrec.optimizer import build_optimizer
from openrec.postprocess import build_post_process
from tools.data import build_dataloader
from tools.utils.ckpt import load_ckpt, save_ckpt
from tools.utils.logging import get_logger
from tools.utils.stats import TrainingStats
from tools.utils.utility import AverageMeter

__all__ = ['Trainer']


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


class Trainer(object):

    def __init__(self, cfg, mode='train'):
        self.cfg = cfg.cfg

        self.local_rank = (int(os.environ['LOCAL_RANK'])
                           if 'LOCAL_RANK' in os.environ else 0)
        self.set_device(self.cfg['Global']['device'])
        mode = mode.lower()
        assert mode in [
            'train_eval',
            'train',
            'eval',
            'test',
        ], 'mode should be train, eval and test'
        if torch.cuda.device_count() > 1 and 'train' in mode:
            torch.distributed.init_process_group(backend='nccl')
            torch.cuda.set_device(self.device)
            self.cfg['Global']['distributed'] = True
        else:
            self.cfg['Global']['distributed'] = False
            self.local_rank = 0

        self.cfg['Global']['output_dir'] = self.cfg['Global'].get(
            'output_dir', 'output')
        os.makedirs(self.cfg['Global']['output_dir'], exist_ok=True)

        self.writer = None
        if self.local_rank == 0 and self.cfg['Global'][
                'use_tensorboard'] and 'train' in mode:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(self.cfg['Global']['output_dir'])

        self.logger = get_logger(
            'openrec',
            os.path.join(self.cfg['Global']['output_dir'], 'train.log')
            if 'train' in mode else None,
        )

        cfg.print_cfg(self.logger.info)

        if self.cfg['Global']['device'] == 'gpu' and self.device.type == 'cpu':
            self.logger.info('cuda is not available, auto switch to cpu')

        self.grad_clip_val = self.cfg['Global'].get('grad_clip_val', 0)
        self.all_ema = self.cfg['Global'].get('all_ema', True)
        self.use_ema = self.cfg['Global'].get('use_ema', True)

        self.set_random_seed(self.cfg['Global'].get('seed', 48))

        # build data loader
        self.train_dataloader = None
        if 'train' in mode:
            cfg.save(
                os.path.join(self.cfg['Global']['output_dir'], 'config.yml'),
                self.cfg)
            self.train_dataloader = build_dataloader(self.cfg, 'Train',
                                                     self.logger)
            self.logger.info(
                f'train dataloader has {len(self.train_dataloader)} iters')
        self.valid_dataloader = None
        if 'eval' in mode and self.cfg['Eval']:
            self.valid_dataloader = build_dataloader(self.cfg, 'Eval',
                                                     self.logger)
            self.logger.info(
                f'valid dataloader has {len(self.valid_dataloader)} iters')

        # build post process
        self.post_process_class = build_post_process(self.cfg['PostProcess'],
                                                     self.cfg['Global'])
        # build model
        # for rec algorithm
        char_num = self.post_process_class.get_character_num()
        self.cfg['Architecture']['Decoder']['out_channels'] = char_num

        self.model = build_model(self.cfg['Architecture'])
        self.logger.info(get_parameter_number(model=self.model))
        self.model = self.model.to(self.device)

        if self.local_rank == 0:
            ema_model = build_model(self.cfg['Architecture'])
            self.ema_model = ema_model.to(self.device)
            self.ema_model.eval()

        use_sync_bn = self.cfg['Global'].get('use_sync_bn', False)
        if use_sync_bn:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                self.model)
            self.logger.info('convert_sync_batchnorm')

        # build loss
        self.loss_class = build_loss(self.cfg['Loss'])

        self.optimizer, self.lr_scheduler = None, None
        if self.train_dataloader is not None:
            # build optim
            self.optimizer, self.lr_scheduler = build_optimizer(
                self.cfg['Optimizer'],
                self.cfg['LRScheduler'],
                epochs=self.cfg['Global']['epoch_num'],
                step_each_epoch=len(self.train_dataloader),
                model=self.model,
            )

        self.eval_class = build_metric(self.cfg['Metric'])

        self.status = load_ckpt(self.model, self.cfg, self.optimizer,
                                self.lr_scheduler)

        if self.cfg['Global']['distributed']:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, [self.local_rank], find_unused_parameters=False)

        # amp
        self.scaler = (torch.cuda.amp.GradScaler() if self.cfg['Global'].get(
            'use_amp', False) else None)

        self.logger.info(
            f'run with torch {torch.__version__} and device {self.device}')

    def load_params(self, params):
        self.model.load_state_dict(params)

    def set_random_seed(self, seed):
        torch.manual_seed(seed)  # 为CPU设置随机种子
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
            torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
        random.seed(seed)
        np.random.seed(seed)

    def set_device(self, device):
        if device == 'gpu' and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.local_rank}')
        else:
            device = torch.device('cpu')
        self.device = device

    def train(self):
        cal_metric_during_train = self.cfg['Global'].get(
            'cal_metric_during_train', False)
        log_smooth_window = self.cfg['Global']['log_smooth_window']
        epoch_num = self.cfg['Global']['epoch_num']
        print_batch_step = self.cfg['Global']['print_batch_step']
        eval_epoch_step = self.cfg['Global'].get('eval_epoch_step', 1)

        start_eval_epoch = 0
        if self.valid_dataloader is not None:
            if type(eval_epoch_step) == list and len(eval_epoch_step) >= 2:
                start_eval_epoch = eval_epoch_step[0]
                eval_epoch_step = eval_epoch_step[1]
                if len(self.valid_dataloader) == 0:
                    start_eval_epoch = 1e111
                    self.logger.info(
                        'No Images in eval dataset, evaluation during training will be disabled'
                    )
                self.logger.info(
                    f'During the training process, after the {start_eval_epoch}th epoch, '
                    f'an evaluation is run every {eval_epoch_step} epoch')
        else:
            start_eval_epoch = 1e111

        eval_batch_step = self.cfg['Global']['eval_batch_step']

        global_step = self.status.get('global_step', 0)

        start_eval_step = 0
        if type(eval_batch_step) == list and len(eval_batch_step) >= 2:
            start_eval_step = eval_batch_step[0]
            eval_batch_step = eval_batch_step[1]
            if len(self.valid_dataloader) == 0:
                self.logger.info(
                    'No Images in eval dataset, evaluation during training '
                    'will be disabled')
                start_eval_step = 1e111
            self.logger.info(
                'During the training process, after the {}th iteration, '
                'an evaluation is run every {} iterations'.format(
                    start_eval_step, eval_batch_step))

        start_epoch = self.status.get('epoch', 1)
        best_metric = self.status.get('metrics', {})
        if self.eval_class.main_indicator not in best_metric:
            best_metric[self.eval_class.main_indicator] = 0
        ema_best_metric = self.status.get('metrics', {})
        ema_best_metric[self.eval_class.main_indicator] = 0
        train_stats = TrainingStats(log_smooth_window, ['lr'])
        self.model.train()

        total_samples = 0
        train_reader_cost = 0.0
        train_batch_cost = 0.0
        best_iter = 0
        ema_stpe = 1
        ema_eval_iter = 0
        loss_avg = 0.
        reader_start = time.time()
        eta_meter = AverageMeter()

        for epoch in range(start_epoch, epoch_num + 1):
            if self.train_dataloader.dataset.need_reset:
                self.train_dataloader = build_dataloader(
                    self.cfg,
                    'Train',
                    self.logger,
                    epoch=epoch % 20 if epoch % 20 != 0 else 20,
                )

            for idx, batch in enumerate(self.train_dataloader):
                batch = [t.to(self.device) for t in batch]
                self.optimizer.zero_grad()
                train_reader_cost += time.time() - reader_start
                # use amp
                if self.scaler:
                    with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                        preds = self.model(batch[0], data=batch[1:])
                        loss = self.loss_class(preds, batch)
                    self.scaler.scale(loss['loss']).backward()
                    if self.grad_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.grad_clip_val)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    preds = self.model(batch[0], data=batch[1:])
                    loss = self.loss_class(preds, batch)
                    avg_loss = loss['loss']
                    avg_loss.backward()
                    if self.grad_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.grad_clip_val)
                    self.optimizer.step()

                if cal_metric_during_train:  # only rec and cls need
                    post_result = self.post_process_class(preds,
                                                          batch,
                                                          training=True)
                    self.eval_class(post_result, batch, training=True)
                    metric = self.eval_class.get_metric()
                    train_stats.update(metric)

                train_batch_time = time.time() - reader_start
                train_batch_cost += train_batch_time
                eta_meter.update(train_batch_time)
                global_step += 1
                total_samples += len(batch[0])

                self.lr_scheduler.step()

                if self.local_rank == 0 and self.use_ema and epoch > (
                        epoch_num - epoch_num // 10):
                    with torch.no_grad():
                        loss_currn = loss['loss'].detach().cpu().numpy().mean()
                        loss_avg = ((loss_avg *
                                     (ema_stpe - 1)) + loss_currn) / (ema_stpe)
                        if ema_stpe == 1:

                            # current_weight  = copy.deepcopy(self.model.module.state_dict())
                            ema_state_dict = copy.deepcopy(
                                self.model.module.state_dict() if self.
                                cfg['Global']['distributed'] else self.model.
                                state_dict())
                            self.ema_model.load_state_dict(ema_state_dict)
                        # if global_step > (epoch_num - epoch_num//10)*max_iter:
                        elif loss_currn <= loss_avg or self.all_ema:
                            # eval_batch_step = 500
                            current_weight = copy.deepcopy(
                                self.model.module.state_dict() if self.
                                cfg['Global']['distributed'] else self.model.
                                state_dict())
                            k1 = 1 / (ema_stpe + 1)
                            k2 = 1 - k1
                            for k, v in ema_state_dict.items():
                                # v = (v * (ema_stpe - 1) + current_weight[k])/ema_stpe
                                v = v * k2 + current_weight[k] * k1
                                # v.req = True
                                ema_state_dict[k] = v
                            # ema_stpe += 1
                            self.ema_model.load_state_dict(ema_state_dict)
                    ema_stpe += 1
                    if global_step > start_eval_step and (
                            global_step -
                            start_eval_step) % eval_batch_step == 0:
                        ema_cur_metric = self.eval_ema()
                        ema_cur_metric_str = f"cur ema metric, {', '.join(['{}: {}'.format(k, v) for k, v in ema_cur_metric.items()])}"
                        self.logger.info(ema_cur_metric_str)
                        state = {
                            'epoch': epoch,
                            'global_step': global_step,
                            'state_dict': self.ema_model.state_dict(),
                            'optimizer': None,
                            'scheduler': None,
                            'config': self.cfg,
                            'metrics': ema_cur_metric,
                        }
                        save_path = os.path.join(
                            self.cfg['Global']['output_dir'],
                            'ema_' + str(ema_eval_iter) + '.pth')
                        torch.save(state, save_path)
                        self.logger.info(f'save ema ckpt to {save_path}')
                        ema_eval_iter += 1
                        if ema_cur_metric[self.eval_class.
                                          main_indicator] >= ema_best_metric[
                                              self.eval_class.main_indicator]:
                            ema_best_metric.update(ema_cur_metric)
                            ema_best_metric['best_epoch'] = epoch
                        best_ema_str = f"best metric, {', '.join(['{}: {}'.format(k, v) for k, v in ema_best_metric.items()])}"
                        self.logger.info(best_ema_str)

                # logger
                stats = {
                    k: float(v)
                    if v.shape == [] else v.detach().cpu().numpy().mean()
                    for k, v in loss.items()
                }
                stats['lr'] = self.lr_scheduler.get_last_lr()[0]
                train_stats.update(stats)

                if self.writer is not None:
                    for k, v in train_stats.get().items():
                        self.writer.add_scalar(f'TRAIN/{k}', v, global_step)

                if self.local_rank == 0 and (
                    (global_step > 0 and global_step % print_batch_step == 0)
                        or (idx >= len(self.train_dataloader) - 1)):
                    logs = train_stats.log()

                    eta_sec = (
                        (epoch_num + 1 - epoch) * len(self.train_dataloader) -
                        idx - 1) * eta_meter.avg
                    eta_sec_format = str(
                        datetime.timedelta(seconds=int(eta_sec)))
                    strs = (
                        f'epoch: [{epoch}/{epoch_num}], global_step: {global_step}, {logs}, '
                        f'avg_reader_cost: {train_reader_cost / print_batch_step:.5f} s, '
                        f'avg_batch_cost: {train_batch_cost / print_batch_step:.5f} s, '
                        f'avg_samples: {total_samples / print_batch_step}, '
                        f'ips: {total_samples / train_batch_cost:.5f} samples/s, '
                        f'eta: {eta_sec_format}')
                    self.logger.info(strs)
                    total_samples = 0
                    train_reader_cost = 0.0
                    train_batch_cost = 0.0
                reader_start = time.time()
                # eval
                if (global_step > start_eval_step and
                    (global_step - start_eval_step) % eval_batch_step
                        == 0) and self.local_rank == 0:
                    cur_metric = self.eval()
                    cur_metric_str = f"cur metric, {', '.join(['{}: {}'.format(k, v) for k, v in cur_metric.items()])}"
                    self.logger.info(cur_metric_str)

                    # logger metric
                    if self.writer is not None:
                        for k, v in cur_metric.items():
                            if isinstance(v, (float, int)):
                                self.writer.add_scalar(f'EVAL/{k}',
                                                       cur_metric[k],
                                                       global_step)

                    if (cur_metric[self.eval_class.main_indicator] >=
                            best_metric[self.eval_class.main_indicator]):
                        best_metric.update(cur_metric)
                        best_metric['best_epoch'] = epoch
                        if self.writer is not None:
                            self.writer.add_scalar(
                                f'EVAL/best_{self.eval_class.main_indicator}',
                                best_metric[self.eval_class.main_indicator],
                                global_step,
                            )
                        if epoch > (epoch_num - epoch_num // 10 - 2):
                            save_ckpt(self.model,
                                      self.cfg,
                                      self.optimizer,
                                      self.lr_scheduler,
                                      epoch,
                                      global_step,
                                      best_metric,
                                      is_best=True,
                                      prefix='best_' + str(best_iter))
                            best_iter += 1
                        # else:
                        save_ckpt(self.model,
                                  self.cfg,
                                  self.optimizer,
                                  self.lr_scheduler,
                                  epoch,
                                  global_step,
                                  best_metric,
                                  is_best=True,
                                  prefix=None)
                    best_str = f"best metric, {', '.join(['{}: {}'.format(k, v) for k, v in best_metric.items()])}"
                    self.logger.info(best_str)
            if self.local_rank == 0 and epoch > start_eval_epoch and (
                    epoch - start_eval_epoch) % eval_epoch_step == 0:
                cur_metric = self.eval()
                cur_metric_str = f"cur metric, {', '.join(['{}: {}'.format(k, v) for k, v in cur_metric.items()])}"
                self.logger.info(cur_metric_str)

                # logger metric
                if self.writer is not None:
                    for k, v in cur_metric.items():
                        if isinstance(v, (float, int)):
                            self.writer.add_scalar(f'EVAL/{k}', cur_metric[k],
                                                   global_step)

                if (cur_metric[self.eval_class.main_indicator] >=
                        best_metric[self.eval_class.main_indicator]):
                    best_metric.update(cur_metric)
                    best_metric['best_epoch'] = epoch
                    if self.writer is not None:
                        self.writer.add_scalar(
                            f'EVAL/best_{self.eval_class.main_indicator}',
                            best_metric[self.eval_class.main_indicator],
                            global_step,
                        )
                    if epoch > (epoch_num - epoch_num // 10 - 2):
                        save_ckpt(self.model,
                                  self.cfg,
                                  self.optimizer,
                                  self.lr_scheduler,
                                  epoch,
                                  global_step,
                                  best_metric,
                                  is_best=True,
                                  prefix='best_' + str(best_iter))
                        best_iter += 1
                    # else:
                    save_ckpt(self.model,
                              self.cfg,
                              self.optimizer,
                              self.lr_scheduler,
                              epoch,
                              global_step,
                              best_metric,
                              is_best=True,
                              prefix=None)
                best_str = f"best metric, {', '.join(['{}: {}'.format(k, v) for k, v in best_metric.items()])}"
                self.logger.info(best_str)

            if self.local_rank == 0:
                save_ckpt(self.model,
                          self.cfg,
                          self.optimizer,
                          self.lr_scheduler,
                          epoch,
                          global_step,
                          best_metric,
                          is_best=False,
                          prefix=None)
                if epoch > (epoch_num - epoch_num // 10 - 2):
                    save_ckpt(self.model,
                              self.cfg,
                              self.optimizer,
                              self.lr_scheduler,
                              epoch,
                              global_step,
                              best_metric,
                              is_best=False,
                              prefix='epoch_' + str(epoch))
                if self.use_ema and epoch > (epoch_num - epoch_num // 10):
                    # if global_step > start_eval_step and (global_step - start_eval_step) % eval_batch_step == 0:
                    ema_cur_metric = self.eval_ema()
                    ema_cur_metric_str = f"cur ema metric, {', '.join(['{}: {}'.format(k, v) for k, v in ema_cur_metric.items()])}"
                    self.logger.info(ema_cur_metric_str)
                    state = {
                        'epoch': epoch,
                        'global_step': global_step,
                        'state_dict': self.ema_model.state_dict(),
                        'optimizer': None,
                        'scheduler': None,
                        'config': self.cfg,
                        'metrics': ema_cur_metric,
                    }
                    save_path = os.path.join(
                        self.cfg['Global']['output_dir'],
                        'ema_' + str(ema_eval_iter) + '.pth')
                    torch.save(state, save_path)
                    self.logger.info(f'save ema ckpt to {save_path}')
                    ema_eval_iter += 1
                    if (ema_cur_metric[self.eval_class.main_indicator] >=
                            ema_best_metric[self.eval_class.main_indicator]):
                        ema_best_metric.update(ema_cur_metric)
                        ema_best_metric['best_epoch'] = epoch
                        # ema_cur_metric_str = f"best ema metric, {', '.join(['{}: {}'.format(k, v) for k, v in ema_best_metric.items()])}"
                    best_ema_str = f"best metric, {', '.join(['{}: {}'.format(k, v) for k, v in ema_best_metric.items()])}"
                    self.logger.info(best_ema_str)
        best_str = f"best metric, {', '.join(['{}: {}'.format(k, v) for k, v in best_metric.items()])}"
        self.logger.info(best_str)
        if self.writer is not None:
            self.writer.close()
        if torch.cuda.device_count() > 1:
            torch.distributed.destroy_process_group()

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            total_frame = 0.0
            total_time = 0.0
            pbar = tqdm(
                total=len(self.valid_dataloader),
                desc='eval model:',
                position=0,
                leave=True,
            )
            sum_images = 0
            for idx, batch in enumerate(self.valid_dataloader):
                batch = [t.to(self.device) for t in batch]
                start = time.time()
                if self.scaler:
                    with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                        preds = self.model(batch[0], data=batch[1:])
                else:
                    preds = self.model(batch[0], data=batch[1:])

                total_time += time.time() - start
                # Obtain usable results from post-processing methods
                # Evaluate the results of the current batch
                post_result = self.post_process_class(preds, batch)
                self.eval_class(post_result, batch)

                pbar.update(1)
                total_frame += len(batch[0])
                sum_images += 1
            # Get final metric，eg. acc or hmean
            metric = self.eval_class.get_metric()

        pbar.close()
        self.model.train()
        metric['fps'] = total_frame / total_time
        return metric

    def eval_ema(self):
        # self.model.eval()
        with torch.no_grad():
            total_frame = 0.0
            total_time = 0.0
            pbar = tqdm(
                total=len(self.valid_dataloader),
                desc='eval ema_model:',
                position=0,
                leave=True,
            )
            sum_images = 0
            for idx, batch in enumerate(self.valid_dataloader):
                batch = [t.to(self.device) for t in batch]
                start = time.time()
                if self.scaler:
                    with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                        preds = self.ema_model(batch[0], data=batch[1:])
                else:
                    preds = self.ema_model(batch[0], data=batch[1:])

                total_time += time.time() - start
                # Obtain usable results from post-processing methods
                # Evaluate the results of the current batch
                post_result = self.post_process_class(preds, batch)
                self.eval_class(post_result, batch)

                pbar.update(1)
                total_frame += len(batch[0])
                sum_images += 1
            # Get final metric，eg. acc or hmean
            metric = self.eval_class.get_metric()

        pbar.close()
        # self.model.train()
        metric['fps'] = total_frame / total_time
        return metric

    def test_dataloader(self):
        starttime = time.time()
        count = 0
        try:
            for data in self.train_dataloader:
                count += 1
                if count % 1 == 0:
                    batch_time = time.time() - starttime
                    starttime = time.time()
                    self.logger.info(
                        f'reader: {count}, {data[0].shape}, {batch_time}')
        except:
            import traceback

            self.logger.info(traceback.format_exc())
        self.logger.info(f'finish reader: {count}, Success!')
