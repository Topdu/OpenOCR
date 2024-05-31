import datetime
import os
import random
import time

import numpy as np
import paddle
import paddle.distributed as dist
from tqdm import tqdm

from openrec.losses import build_loss
from openrec.metrics import build_metric
from openrec.modeling import build_model
from openrec.optimizer import build_optimizer
from openrec.postprocess import build_post_process

from tools.data import build_dataloader
from tools.utils.save_load import save_model, load_model
from tools.utils.logging import get_logger
from tools.utils.stats import TrainingStats
from tools.utils.utility import AverageMeter

dist.get_world_size()

__all__ = ["Trainer"]


def to_float32(preds):
    if isinstance(preds, dict):
        for k in preds:
            if isinstance(preds[k], dict) or isinstance(preds[k], list):
                preds[k] = to_float32(preds[k])
            elif isinstance(preds[k], paddle.Tensor):
                preds[k] = preds[k].astype(paddle.float32)
    elif isinstance(preds, list):
        for k in range(len(preds)):
            if isinstance(preds[k], dict):
                preds[k] = to_float32(preds[k])
            elif isinstance(preds[k], list):
                preds[k] = to_float32(preds[k])
            elif isinstance(preds[k], paddle.Tensor):
                preds[k] = preds[k].astype(paddle.float32)
    elif isinstance(preds, paddle.Tensor):
        preds = preds.astype(paddle.float32)
    return preds


class Trainer(object):
    def __init__(self, cfg, mode="train"):
        self.cfg = cfg.cfg

        # self.set_device(self.cfg["Global"]["device"])
        use_gpu = self.cfg["Global"]["use_gpu"]
        self.device = 'gpu:{}'.format(dist.ParallelEnv()
                                      .dev_id) if use_gpu else 'cpu'
        device = paddle.set_device(self.device)

        if dist.get_world_size() != 1:
            self.cfg["Global"]["distributed"] = True
        else:
            self.cfg["Global"]["distributed"] = False
            self.local_rank = 0
        if self.cfg['Global']['distributed']:
            dist.init_parallel_env()
            self.local_rank = dist.get_rank()
        self.cfg["Global"]["output_dir"] = self.cfg["Global"].get("output_dir",
                                                                  "output")
        os.makedirs(self.cfg["Global"]["output_dir"], exist_ok=True)

        self.writer = None

        self.logger = get_logger(
            "openrec",
            os.path.join(self.cfg["Global"]["output_dir"], "train.log")
            if "train" in mode else None, )

        cfg.print_cfg(self.logger.info)

        self.set_random_seed(self.cfg["Global"].get("seed", 48))

        mode = mode.lower()
        assert mode in [
            "train_eval",
            "train",
            "eval",
            "test",
        ], "mode should be train, eval and test"

        # build data loader
        self.train_dataloader = None
        if "train" in mode:
            cfg.save(
                os.path.join(self.cfg["Global"]["output_dir"], "config.yml"),
                self.cfg)
            self.train_dataloader = build_dataloader(self.cfg, "Train", device,
                                                     self.logger)
            self.logger.info(
                f"train dataloader has {len(self.train_dataloader)} iters")
        self.valid_dataloader = None
        if "eval" in mode and self.cfg["Eval"]:
            self.valid_dataloader = build_dataloader(self.cfg, "Eval", device,
                                                     self.logger)
            self.logger.info(
                f"valid dataloader has {len(self.valid_dataloader)} iters")

        # build post process
        self.post_process_class = build_post_process(self.cfg["PostProcess"])
        # build model
        # for rec algorithm
        char_num = len(getattr(self.post_process_class, "character"))
        self.cfg["Architecture"]["Decoder"]["out_channels"] = char_num

        self.model = build_model(self.cfg["Architecture"])
        self.model = self.model.to(self.device)
        use_sync_bn = self.cfg["Global"].get("use_sync_bn", False)
        if use_sync_bn:
            self.model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(
                self.model)
            self.logger.info("convert_sync_batchnorm")

        # build loss
        self.loss_class = build_loss(self.cfg["Loss"])

        self.optimizer, self.lr_scheduler = None, None
        if self.train_dataloader is not None:
            # build optim
            self.optimizer, self.lr_scheduler = build_optimizer(
                self.cfg['Optimizer'],
                epochs=self.cfg['Global']['epoch_num'],
                step_each_epoch=len(self.train_dataloader),
                model=self.model)

        self.eval_class = build_metric(self.cfg["Metric"])

        # self.status = load_ckpt(self.model, self.cfg, self.optimizer,
        #                         self.lr_scheduler)

        # amp
        use_amp = self.cfg["Global"].get("use_amp", False)
        self.amp_level = self.cfg["Global"].get("amp_level", 'O2')
        self.amp_dtype = self.cfg["Global"].get("amp_dtype", 'float16')
        self.amp_custom_black_list = self.cfg['Global'].get(
            'amp_custom_black_list', [])
        self.amp_custom_white_list = self.cfg['Global'].get(
            'amp_custom_white_list', [])
        if use_amp:
            AMP_RELATED_FLAGS_SETTING = {'FLAGS_max_inplace_grad_add': 8, }
            if paddle.is_compiled_with_cuda():
                AMP_RELATED_FLAGS_SETTING.update({
                    'FLAGS_cudnn_batchnorm_spatial_persistent': 1,
                    'FLAGS_gemm_use_half_precision_compute_type': 0,
                })
            paddle.set_flags(AMP_RELATED_FLAGS_SETTING)
            scale_loss = self.cfg["Global"].get("scale_loss", 1.0)
            use_dynamic_loss_scaling = self.cfg["Global"].get(
                "use_dynamic_loss_scaling", False)
            self.scaler = paddle.amp.GradScaler(
                init_loss_scaling=scale_loss,
                use_dynamic_loss_scaling=use_dynamic_loss_scaling)
            if self.amp_level == "O2":
                self.model, self.optimizer = paddle.amp.decorate(
                    models=self.model,
                    optimizers=self.optimizer,
                    level=self.amp_level,
                    master_weight=True,
                    dtype=self.amp_dtype)
        else:
            self.scaler = None
        self.status = load_model(self.cfg, self.model, self.optimizer,
                                 self.cfg['Architecture']["model_type"])
        if self.cfg['Global']['distributed']:
            self.model = paddle.DataParallel(self.model)

        self.logger.info(
            f"run with paddle {paddle.__version__} and device {self.device}")

    def set_random_seed(self, seed):
        paddle.seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    # def set_device(self, device):
    #     if device == "gpu" and torch.cuda.is_available():
    #         device = torch.device(f"cuda:{self.local_rank}")
    #     else:
    #         device = torch.device("cpu")
    #     self.device = device

    def train(self):
        cal_metric_during_train = self.cfg["Global"].get(
            "cal_metric_during_train", False)
        log_smooth_window = self.cfg["Global"]["log_smooth_window"]
        epoch_num = self.cfg["Global"]["epoch_num"]
        print_batch_step = self.cfg["Global"]["print_batch_step"]
        eval_epoch_step = self.cfg["Global"].get("eval_epoch_step", 1)

        start_eval_epoch = 0
        if self.valid_dataloader is not None:
            if type(eval_epoch_step) == list and len(eval_epoch_step) >= 2:
                start_eval_epoch = eval_epoch_step[0]
                eval_epoch_step = eval_epoch_step[1]
                if len(self.valid_dataloader) == 0:
                    start_eval_epoch = 1e111
                    self.logger.info(
                        "No Images in eval dataset, evaluation during training will be disabled"
                    )
                self.logger.info(
                    f"During the training process, after the {start_eval_epoch}th epoch, "
                    f"an evaluation is run every {eval_epoch_step} epoch")
        else:
            start_eval_epoch = 1e111

        eval_batch_step = self.cfg["Global"]["eval_batch_step"]

        global_step = self.status.get("global_step", 0)

        start_eval_step = 0
        if type(eval_batch_step) == list and len(eval_batch_step) >= 2:
            start_eval_step = eval_batch_step[0]
            eval_batch_step = eval_batch_step[1]
            if len(self.valid_dataloader) == 0:
                self.logger.info(
                    "No Images in eval dataset, evaluation during training "
                    "will be disabled")
                start_eval_step = 1e111
            self.logger.info(
                "During the training process, after the {}th iteration, "
                "an evaluation is run every {} iterations".format(
                    start_eval_step, eval_batch_step))

        start_epoch = self.status.get("epoch", 1)
        best_metric = self.status.get("metrics", {})
        if self.eval_class.main_indicator not in best_metric:
            best_metric[self.eval_class.main_indicator] = 0
        train_stats = TrainingStats(log_smooth_window, ["lr"])
        self.model.train()

        total_samples = 0
        train_reader_cost = 0.0
        train_batch_cost = 0.0
        reader_start = time.time()
        eta_meter = AverageMeter()

        for epoch in range(start_epoch, epoch_num + 1):
            if self.train_dataloader.dataset.need_reset:
                self.train_dataloader = build_dataloader(
                    self.cfg,
                    "Train",
                    self.logger,
                    epoch=epoch % 20 if epoch % 20 != 0 else 20, )

            for idx, batch in enumerate(self.train_dataloader):
                # batch = [t.to(self.device) for t in batch]

                train_reader_cost += time.time() - reader_start
                # use amp
                if self.scaler:
                    # with torch.cuda.amp.autocast():
                    #     preds = self.model(batch[0], data=batch[1:])
                    #     loss = self.loss_class(preds, batch)
                    # self.scaler.scale(loss["loss"]).backward()
                    # self.scaler.step(self.optimizer)
                    # self.scaler.update()
                    with paddle.amp.auto_cast(
                            level=self.amp_level,
                            custom_black_list=self.amp_custom_black_list,
                            custom_white_list=self.amp_custom_white_list,
                            dtype=self.amp_dtype):
                        preds = self.model(batch[0], data=batch[1:])
                    preds = to_float32(preds)
                    loss = self.loss_class(preds, batch)
                    avg_loss = loss['loss']
                    scaled_avg_loss = self.scaler.scale(avg_loss)
                    scaled_avg_loss.backward()
                    self.scaler.minimize(self.optimizer, scaled_avg_loss)
                else:
                    preds = self.model(batch[0], data=batch[1:])
                    loss = self.loss_class(preds, batch)
                    avg_loss = loss["loss"]
                    avg_loss.backward()
                    self.optimizer.step()
                self.optimizer.clear_grad()
                if cal_metric_during_train:  # only rec and cls need
                    post_result = self.post_process_class(preds, batch)
                    self.eval_class(post_result, batch)
                    metric = self.eval_class.get_metric()
                    train_stats.update(metric)

                train_batch_time = time.time() - reader_start
                train_batch_cost += train_batch_time
                eta_meter.update(train_batch_time)
                global_step += 1
                total_samples += len(batch[0])

                self.lr_scheduler.step()

                # logger
                stats = {
                    k: float(v)
                    if v.shape == [] else v.detach().cpu().numpy().mean()
                    for k, v in loss.items()
                }
                stats["lr"] = self.optimizer.get_lr()
                train_stats.update(stats)

                if self.writer is not None:
                    for k, v in train_stats.get().items():
                        self.writer.add_scalar(f"TRAIN/{k}", v, global_step)

                if self.local_rank == 0 and (
                    (global_step > 0 and global_step % print_batch_step == 0) or
                    (idx >= len(self.train_dataloader) - 1)):
                    logs = train_stats.log()

                    eta_sec = (
                        (epoch_num + 1 - epoch) * len(self.train_dataloader) -
                        idx - 1) * eta_meter.avg
                    eta_sec_format = str(
                        datetime.timedelta(seconds=int(eta_sec)))
                    strs = (
                        f"epoch: [{epoch}/{epoch_num}], global_step: {global_step}, {logs}, "
                        f"avg_reader_cost: {train_reader_cost / print_batch_step:.5f} s, "
                        f"avg_batch_cost: {train_batch_cost / print_batch_step:.5f} s, "
                        f"avg_samples: {total_samples / print_batch_step}, "
                        f"ips: {total_samples / train_batch_cost:.5f} samples/s, "
                        f"eta: {eta_sec_format}")
                    self.logger.info(strs)
                    total_samples = 0
                    train_reader_cost = 0.0
                    train_batch_cost = 0.0
                reader_start = time.time()
                # eval
                # print(global_step, start_eval_step, eval_batch_step, ((global_step > start_eval_step and (global_step - start_eval_step) % eval_batch_step == 0) and self.local_rank == 0))
                if (global_step > start_eval_step and
                    (global_step - start_eval_step
                     ) % eval_batch_step == 0) and self.local_rank == 0:
                    # or (self.local_rank == 0 and epoch > start_eval_epoch and (epoch - start_eval_epoch) % eval_epoch_step == 0 and epoch>1):
                    cur_metric = self.eval()
                    cur_metric_str = f"cur metric, {', '.join(['{}: {}'.format(k, v) for k, v in cur_metric.items()])}"
                    self.logger.info(cur_metric_str)

                    # logger metric
                    if self.writer is not None:
                        for k, v in cur_metric.items():
                            if isinstance(v, (float, int)):
                                self.writer.add_scalar(
                                    f"EVAL/{k}", cur_metric[k], global_step)

                    if (cur_metric[self.eval_class.main_indicator] >=
                            best_metric[self.eval_class.main_indicator]):
                        best_metric.update(cur_metric)
                        best_metric["best_epoch"] = epoch
                        if self.writer is not None:
                            self.writer.add_scalar(
                                f"EVAL/best_{self.eval_class.main_indicator}",
                                best_metric[self.eval_class.main_indicator],
                                global_step, )
                        save_model(
                            self.model,
                            self.optimizer,
                            self.cfg["Global"]["output_dir"],
                            self.logger,
                            self.cfg,
                            is_best=True,
                            prefix='best_accuracy',
                            best_model_dict=best_metric,
                            epoch=epoch,
                            global_step=global_step)
                    best_str = f"best metric, {', '.join(['{}: {}'.format(k, v) for k, v in best_metric.items()])}"
                    self.logger.info(best_str)
            if (self.local_rank == 0 and epoch > start_eval_epoch and
                (epoch - start_eval_epoch) % eval_epoch_step == 0):
                cur_metric = self.eval()
                cur_metric_str = f"cur metric, {', '.join(['{}: {}'.format(k, v) for k, v in cur_metric.items()])}"
                self.logger.info(cur_metric_str)

                # logger metric
                if self.writer is not None:
                    for k, v in cur_metric.items():
                        if isinstance(v, (float, int)):
                            self.writer.add_scalar(f"EVAL/{k}", cur_metric[k],
                                                   global_step)

                if (cur_metric[self.eval_class.main_indicator] >=
                        best_metric[self.eval_class.main_indicator]):
                    best_metric.update(cur_metric)
                    best_metric["best_epoch"] = epoch
                    if self.writer is not None:
                        self.writer.add_scalar(
                            f"EVAL/best_{self.eval_class.main_indicator}",
                            best_metric[self.eval_class.main_indicator],
                            global_step, )
                    save_model(
                        self.model,
                        self.optimizer,
                        self.cfg["Global"]["output_dir"],
                        self.logger,
                        self.cfg,
                        is_best=True,
                        prefix='best_accuracy',
                        best_model_dict=best_metric,
                        epoch=epoch,
                        global_step=global_step)
                best_str = f"best metric, {', '.join(['{}: {}'.format(k, v) for k, v in best_metric.items()])}"
                self.logger.info(best_str)

            if self.local_rank == 0:
                save_model(
                    self.model,
                    self.optimizer,
                    self.cfg["Global"]["output_dir"],
                    self.logger,
                    self.cfg,
                    is_best=True,
                    prefix='latest',
                    best_model_dict=best_metric,
                    epoch=epoch,
                    global_step=global_step)
        best_str = f"best metric, {', '.join(['{}: {}'.format(k, v) for k, v in best_metric.items()])}"
        self.logger.info(best_str)
        if self.writer is not None:
            self.writer.close()
        # if torch.cuda.device_count() > 1:
        #     torch.distributed.destroy_process_group()

    def eval(self):
        self.model.eval()
        with paddle.no_grad():
            total_frame = 0.0
            total_time = 0.0
            pbar = tqdm(
                total=len(self.valid_dataloader),
                desc="eval model:",
                position=0,
                leave=True, )
            sum_images = 0
            for idx, batch in enumerate(self.valid_dataloader):
                # batch = [t.to(self.device) for t in batch]
                start = time.time()
                if self.scaler:
                    with paddle.amp.auto_cast(
                            level=self.amp_level,
                            custom_black_list=self.amp_custom_black_list):
                        preds = self.model(batch[0], data=batch[1:])
                    preds = to_float32(preds)
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
        metric["fps"] = total_frame / total_time
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
                        f"reader: {count}, {data[0].shape}, {batch_time}")
        except:
            import traceback

            self.logger.info(traceback.format_exc())
        self.logger.info(f"finish reader: {count}, Success!")
