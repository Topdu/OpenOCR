import csv
import os
import sys
import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tools.data import build_dataloader
from tools.engine.config import Config
from tools.engine.trainer import Trainer
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

    msr = False
    if 'RatioDataSet' in cfg.cfg['Eval']['dataset']['name']:
        msr = True

    if cfg.cfg['Global']['output_dir'][-1] == '/':
        cfg.cfg['Global']['output_dir'] = cfg.cfg['Global']['output_dir'][:-1]
    if cfg.cfg['Global']['pretrained_model'] is None:
        cfg.cfg['Global'][
            'pretrained_model'] = cfg.cfg['Global']['output_dir'] + '/best.pth'
    cfg.cfg['Global']['use_amp'] = False
    cfg.cfg['PostProcess']['with_ratio'] = True
    cfg.cfg['Metric']['with_ratio'] = True
    cfg.cfg['Metric']['max_len'] = 25
    cfg.cfg['Metric']['max_ratio'] = 12
    cfg.cfg['Eval']['dataset']['transforms'][-1]['KeepKeys'][
        'keep_keys'].append('real_ratio')
    trainer = Trainer(cfg, mode='eval')

    best_model_dict = trainer.status.get('metrics', {})
    trainer.logger.info('metric in ckpt ***************')
    for k, v in best_model_dict.items():
        trainer.logger.info('{}:{}'.format(k, v))

    data_dirs_list = [
        [
            '../test/IIIT5k/', '../test/SVT/', '../test/IC13_857/',
            '../test/IC15_1811/', '../test/SVTP/', '../test/CUTE80/'
        ],
        [
            '../u14m/curve/', '../u14m/multi_oriented/', '../u14m/artistic/',
            '../u14m/contextless/', '../u14m/salient/', '../u14m/multi_words/',
            '../u14m/general/'
        ], ['../OST/weak/', '../OST/heavy/'],
        ['../wordart_test/', '../test/IC13_1015/', '../test/IC15_2077/']
    ]
    cfg = cfg.cfg
    file_csv = open(
        cfg['Global']['output_dir'] + '/' +
        cfg['Global']['output_dir'].split('/')[-1] +
        '_eval_all_length_ratio.csv', 'w')
    csv_w = csv.writer(file_csv)
    cfg['Eval']['dataset']['name'] = cfg['Eval']['dataset']['name'] + 'Test'
    for data_dirs in data_dirs_list:

        acc_each = []
        acc_each_real = []
        acc_each_lower = []
        acc_each_ingore_space = []
        acc_each_ingore_space_lower = []
        acc_each_ignore_space_symbol = []
        acc_each_lower_ignore_space_symbol = []
        acc_each_num = []
        acc_each_dis = []
        each_len = {}
        each_ratio = {}
        for datadir in data_dirs:
            config_each = cfg.copy()
            if msr:
                config_each['Eval']['dataset']['data_dir_list'] = [datadir]
            else:
                config_each['Eval']['dataset']['data_dir'] = datadir
            valid_dataloader = build_dataloader(config_each, 'Eval',
                                                trainer.logger)
            trainer.logger.info(
                f'{datadir} valid dataloader has {len(valid_dataloader)} iters'
            )
            trainer.valid_dataloader = valid_dataloader
            metric = trainer.eval()
            acc_each.append(metric['acc'] * 100)
            acc_each_real.append(metric['acc_real'] * 100)
            acc_each_lower.append(metric['acc_lower'] * 100)
            acc_each_ingore_space.append(metric['acc_ignore_space'] * 100)
            acc_each_ingore_space_lower.append(
                metric['acc_ignore_space_lower'] * 100)
            acc_each_ignore_space_symbol.append(
                metric['acc_ignore_space_symbol'] * 100)
            acc_each_lower_ignore_space_symbol.append(
                metric['acc_ignore_space_lower_symbol'] * 100)
            acc_each_dis.append(metric['norm_edit_dis'])
            acc_each_num.append(metric['num_samples'])

            trainer.logger.info('metric eval ***************')
            csv_w.writerow([datadir])
            for k, v in metric.items():
                trainer.logger.info('{}:{}'.format(k, v))
                if 'each' in k:
                    csv_w.writerow([k] + v)
                    if 'each_len' in k:
                        each_len[k] = each_len.get(k, []) + [np.array(v)]
                    if 'each_ratio' in k:
                        each_ratio[k] = each_ratio.get(k, []) + [np.array(v)]
        data_name = [
            data_n[:-1].split('/')[-1]
            if data_n[-1] == '/' else data_n.split('/')[-1]
            for data_n in data_dirs
        ]
        csv_w.writerow(['-'] + data_name + ['arithmetic_avg'] +
                       ['weighted_avg'])
        csv_w.writerow([''] + acc_each_num)
        avg1 = np.array(acc_each) * np.array(acc_each_num) / sum(acc_each_num)
        csv_w.writerow(['acc'] + acc_each + [sum(acc_each) / len(acc_each)] +
                       [avg1.sum().tolist()])
        print(acc_each + [sum(acc_each) / len(acc_each)] +
              [avg1.sum().tolist()])
        avg1 = np.array(acc_each_dis) * np.array(acc_each_num) / sum(
            acc_each_num)
        csv_w.writerow(['norm_edit_dis'] + acc_each_dis +
                       [sum(acc_each_dis) / len(acc_each)] +
                       [avg1.sum().tolist()])

        avg1 = np.array(acc_each_real) * np.array(acc_each_num) / sum(
            acc_each_num)
        csv_w.writerow(['acc_real'] + acc_each_real +
                       [sum(acc_each_real) / len(acc_each_real)] +
                       [avg1.sum().tolist()])
        avg1 = np.array(acc_each_lower) * np.array(acc_each_num) / sum(
            acc_each_num)
        csv_w.writerow(['acc_lower'] + acc_each_lower +
                       [sum(acc_each_lower) / len(acc_each_lower)] +
                       [avg1.sum().tolist()])
        avg1 = np.array(acc_each_ingore_space) * np.array(acc_each_num) / sum(
            acc_each_num)
        csv_w.writerow(
            ['acc_ignore_space'] + acc_each_ingore_space +
            [sum(acc_each_ingore_space) / len(acc_each_ingore_space)] +
            [avg1.sum().tolist()])
        avg1 = np.array(acc_each_ingore_space_lower) * np.array(
            acc_each_num) / sum(acc_each_num)
        csv_w.writerow(['acc_ignore_space_lower'] +
                       acc_each_ingore_space_lower + [
                           sum(acc_each_ingore_space_lower) /
                           len(acc_each_ingore_space_lower)
                       ] + [avg1.sum().tolist()])
        avg1 = np.array(acc_each_ignore_space_symbol) * np.array(
            acc_each_num) / sum(acc_each_num)
        csv_w.writerow(['acc_ignore_space_symbol'] +
                       acc_each_ignore_space_symbol + [
                           sum(acc_each_ignore_space_symbol) /
                           len(acc_each_ignore_space_symbol)
                       ] + [avg1.sum().tolist()])
        avg1 = np.array(acc_each_lower_ignore_space_symbol) * np.array(
            acc_each_num) / sum(acc_each_num)
        csv_w.writerow(['acc_ignore_space_lower_symbol'] +
                       acc_each_lower_ignore_space_symbol + [
                           sum(acc_each_lower_ignore_space_symbol) /
                           len(acc_each_lower_ignore_space_symbol)
                       ] + [avg1.sum().tolist()])

        sum_all = np.array(each_len['each_len_num']).sum(0)
        for k, v in each_len.items():
            if k != 'each_len_num':
                v_sum_weight = (np.array(v) *
                                np.array(each_len['each_len_num'])).sum(0)
                sum_all_pad = np.where(sum_all == 0, 1., sum_all)
                v_all = v_sum_weight / sum_all_pad
                v_all = np.where(sum_all == 0, 0., v_all)
                csv_w.writerow([k] + v_all.tolist())
            else:
                csv_w.writerow([k] + sum_all.tolist())

        sum_all = np.array(each_ratio['each_ratio_num']).sum(0)
        for k, v in each_ratio.items():
            if k != 'each_ratio_num':
                v_sum_weight = (np.array(v) *
                                np.array(each_ratio['each_ratio_num'])).sum(0)
                sum_all_pad = np.where(sum_all == 0, 1., sum_all)
                v_all = v_sum_weight / sum_all_pad
                v_all = np.where(sum_all == 0, 0., v_all)
                csv_w.writerow([k] + v_all.tolist())
            else:
                csv_w.writerow([k] + sum_all.tolist())

    file_csv.close()


if __name__ == '__main__':
    main()
