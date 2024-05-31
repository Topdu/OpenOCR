import csv
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tools.data import build_dataloader
from tools.engine import Config, Trainer
from tools.utility import ArgsParser


def parse_args():
    parser = ArgsParser()
    args = parser.parse_args()
    return args


def main():
    FLAGS = parse_args()
    cfg = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop('opt')
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)

    cfg.cfg['Global']['use_amp'] = False
    if cfg.cfg['Global']['pretrained_model'] is None:
        cfg.cfg['Global'][
            'pretrained_model'] = cfg.cfg['Global']['output_dir'] + '/best.pth'
    trainer = Trainer(cfg, mode='eval')

    best_model_dict = trainer.status.get('metrics', {})
    trainer.logger.info('metric in ckpt ***************')
    for k, v in best_model_dict.items():
        trainer.logger.info('{}:{}'.format(k, v))

    data_dirs_list = [[
        '../benchmark_bctr/benchmark_bctr_test/scene_test',
        '../benchmark_bctr/benchmark_bctr_test/web_test',
        '../benchmark_bctr/benchmark_bctr_test/document_test',
        '../benchmark_bctr/benchmark_bctr_test/handwriting_test'
    ]]
    cfg = cfg.cfg
    file_csv = open(
        cfg['Global']['output_dir'] + '/' +
        cfg['Global']['output_dir'].split('/')[3] + '_test_ch_all.csv', 'w')
    csv_w = csv.writer(file_csv)
    cfg['Eval']['dataset']['name'] = 'LMDBDataSet'
    for data_dirs in data_dirs_list:

        acc_each = []
        for datadir in data_dirs:
            config_each = cfg.copy()
            config_each['Eval']['dataset']['data_dir'] = datadir
            # config_each['Eval']['dataset']['label_file_list']=[label_file_list]
            valid_dataloader = build_dataloader(config_each, 'Eval',
                                                trainer.logger)
            trainer.logger.info(
                f'{datadir} valid dataloader has {len(valid_dataloader)} iters'
            )
            # valid_dataloaders.append(valid_dataloader)
            trainer.valid_dataloader = valid_dataloader
            metric = trainer.eval()
            acc_each.append(metric['acc'] * 100)

            trainer.logger.info('metric eval ***************')
            for k, v in metric.items():
                trainer.logger.info('{}:{}'.format(k, v))
        csv_w.writerow(acc_each + [sum(acc_each) / len(acc_each)])
        print(acc_each + [sum(acc_each) / len(acc_each)])
    file_csv.close()


if __name__ == '__main__':
    main()
