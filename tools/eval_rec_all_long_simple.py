import csv
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import numpy as np

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
    cfg.cfg['Global']['max_text_length'] = 200
    cfg.cfg['Architecture']['Decoder']['max_len'] = 200
    cfg.cfg['Metric']['name'] = 'RecMetricLong'
    if cfg.cfg['Global']['pretrained_model'] is None:
        cfg.cfg['Global'][
            'pretrained_model'] = cfg.cfg['Global']['output_dir'] + '/best.pth'
    trainer = Trainer(cfg, mode='eval')

    best_model_dict = trainer.status.get('metrics', {})
    trainer.logger.info('metric in ckpt ***************')
    for k, v in best_model_dict.items():
        trainer.logger.info('{}:{}'.format(k, v))

    data_dirs_list = [
        #         ['../test/IC13_857/',
        # '../test/SVT/',
        # '../test/IIIT5k/',
        # '../test/IC15_1811/',
        # '../test/SVTP/',
        # '../test/CUTE80/'],
        # [
        #     # '../test/IC13_857/',
        [
            #     './Union14M-LMDB-LongTest/general/',
            # './Union14M-LMDB-LongTest/multi_words/',
            # './Union14M-LMDB-LongTest/salient/',
            '/data/duyongkun/OpenOCR_igtr/ultra_long/ultra_long_26_35_list.txt',
            '/data/duyongkun/OpenOCR_igtr/ultra_long/ultra_long_36_55_list.txt',
            '/data/duyongkun/OpenOCR_igtr/ultra_long/ultra_long_56_list.txt',
            # './long_val/ctw_1500_crop_long',
            # './long_val/u14m_train'
        ],

        # ['../test/IC13_857/',
        # '../test/SVT/',
        # '../test/IIIT5k/',
        # '../test/IC15_1811/',
        # '../test/SVTP/',
        # '../test/CUTE80/'],

        # ['../u14m/curve/',
        # '../u14m/multi_oriented/',
        # '../u14m/artistic/',
        # '../u14m/contextless/',
        # '../u14m/salient/',
        # '../u14m/multi_words/',
        # '../u14m/general/',
        # ],

        # ['../ha_lmdb/new_stretch_0.5/',
        # '../ha_lmdb/new_stretch_1/',
        # '../ha_lmdb/new_stretch_1.5/',
        # '../ha_lmdb/new_stretch_2/',
        # '../ha_lmdb/new_stretch_2.5/',
        # '../ha_lmdb/new_stretch_3/'],

        # ['../ca_lmdb/new_distort_1/',
        # '../ca_lmdb/new_distort_2/',
        # '../ca_lmdb/new_distort_3/',
        # '../ca_lmdb/new_distort_4/',
        # '../ca_lmdb/new_distort_5/',
        # '../ca_lmdb/new_distort_6/']
        # '/paddle/data/ocr_data/evaluation/IC15_2077/',
        # '/paddle/data/ocr_data/evaluation/IC13_1015/',
        # '/paddle/data/ocr_data/evaluation/IC03_867/',
        # '/paddle/data/ocr_data/evaluation/IC03_860/'
    ]
    # 'IC15_2077/'
    # 'IC13_1015/'
    # 'IC03_867/'
    # 'IC03_860/'
    # print(cfg)
    cfg = cfg.cfg
    cfg['Eval']['dataset']['name'] = 'SimpleDataSet'
    file_csv = open(
        './output/rec/' + cfg['Global']['output_dir'].split('/')[3] +
        '_result1_1_test_all_long_simple_bi_bs1.csv', 'w')
    csv_w = csv.writer(file_csv)

    for data_dirs in data_dirs_list:
        # valid_dataloaders = []
        # dataset_names = []
        acc_each = []
        acc_each_num = []
        acc_each_dis = []
        # acc_each_first = []
        # acc_each_last = []
        # acc_each_len = []
        # first_error_but = []
        # last_error_but = []
        # one_correct_num = []
        # one_error_num = []
        each_long = {}
        for datadir in data_dirs:
            config_each = cfg.copy()
            config_each['Eval']['dataset']['label_file_list'] = [datadir]
            # config_each['Eval']['dataset']['label_file_list']= [datadir] #['/data/duyongkun/OpenOCR_igtr/ultra_long/ultra_long_25_list.txt']
            # if "LongTest" in datadir : #or "test" in datadir:

            #     # print(config_each['Eval']['dataset'])
            # config_each['Eval']['dataset']['transforms'][1]['PARSeqLabelEncode']['max_text_length'] = 70
            #     config_each['Eval']['dataset']['transforms'][2] = {"NextResize": {"image_shape": [3, 32, 128]} }
            #     #   image_shape: [3, 32, 128]
            #     #   padding: False}
            #     config_each['Eval']['loader']['batch_size_per_card'] = 1
            #     config_each['Eval']['loader']['num_workers'] = 1
            #     config_each['Architecture']['Decoder']['next_pred'] = True
            # else:
            #     config_each['Global']['max_text_length'] = 25
            #     config_each['Eval']['dataset']['transforms'][1]['PARSeqLabelEncode']['max_text_length'] = 25
            #     config_each['Eval']['loader']['batch_size_per_card'] = 256
            #     config_each['Eval']['loader']['num_workers'] = 4
            #     config_each['Architecture']['Decoder']['next_pred'] = False
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
            acc_each_dis.append(metric['norm_edit_dis'])
            acc_each_num.append(metric['all_num'])
            # acc_each_first.append(metric['first_acc']*100)
            # acc_each_last.append(metric['last_acc']*100)
            # acc_each_len.append(metric['len_acc']*100)
            # first_error_but.append(metric['first_error_but']*100)
            # last_error_but.append(metric['last_error_but']*100)
            # one_correct_num.append(metric['one_acc']*100)
            # one_error_num.append(metric['one_error']*100)

            trainer.logger.info('metric eval ***************')
            for k, v in metric.items():
                trainer.logger.info('{}:{}'.format(k, v))
                if 'each' in k:
                    csv_w.writerow([k] + v[26:])
                    each_long[k] = each_long.get(k, []) + [np.array(v[26:])]
        avg1 = np.array(acc_each) * np.array(acc_each_num) / sum(acc_each_num)
        csv_w.writerow(acc_each + [avg1.sum().tolist()] +
                       [sum(acc_each) / len(acc_each)])
        print(acc_each + [avg1.sum().tolist()] +
              [sum(acc_each) / len(acc_each)])
        avg1 = np.array(acc_each_dis) * np.array(acc_each_num) / sum(
            acc_each_num)
        csv_w.writerow(acc_each_dis + [avg1.sum().tolist()] +
                       [sum(acc_each_dis) / len(acc_each)])

        # csv_w.writerow(acc_each_first+[sum(acc_each_first)/len(acc_each)])
        # csv_w.writerow(acc_each_last+[sum(acc_each_last)/len(acc_each)])
        # csv_w.writerow(acc_each_len+[sum(acc_each_len)/len(acc_each)])
        # csv_w.writerow(first_error_but+[sum(first_error_but)/len(acc_each)])
        # csv_w.writerow(last_error_but+[sum(last_error_but)/len(acc_each)])
        # csv_w.writerow(one_correct_num+[sum(one_correct_num)/len(acc_each)])
        # csv_w.writerow(one_error_num+[sum(one_error_num)/len(acc_each)])
        sum_all = np.array(each_long['each_len_num']).sum(0)
        for k, v in each_long.items():
            if k != 'each_len_num':
                v_sum_weight = (np.array(v) *
                                np.array(each_long['each_len_num'])).sum(0)
                sum_all_pad = np.where(sum_all == 0, 1., sum_all)
                v_all = v_sum_weight / sum_all_pad
                v_all = np.where(sum_all == 0, 0., v_all)
                csv_w.writerow([k] + v_all.tolist())
                v_26_40 = (v_all[:10] * sum_all[:10]) / sum_all[:10].sum()
                csv_w.writerow([k + '26_35'] + [v_26_40.sum().tolist()] +
                               [sum_all[:10].sum().tolist()])
                v_41_55 = (v_all[10:30] *
                           sum_all[10:30]) / sum_all[10:30].sum()
                csv_w.writerow([k + '36_55'] + [v_41_55.sum().tolist()] +
                               [sum_all[10:30].sum().tolist()])
                v_56_70 = (v_all[30:] * sum_all[30:]) / sum_all[30:].sum()
                csv_w.writerow([k + '56'] + [v_56_70.sum().tolist()] +
                               [sum_all[30:].sum().tolist()])
            else:
                # v = np.array(each_long['each_len_num']).sum(0)
                csv_w.writerow([k] + sum_all.tolist())
        # print(acc_each_first+[sum(acc_each_first)/len(acc_each)])
        # print(acc_each_last+[sum(acc_each_last)/len(acc_each)])
        # print(acc_each_len+[sum(acc_each_len)/len(acc_each)])
        # print(first_error_but+[sum(first_error_but)/len(acc_each)])
        # print(last_error_but+[sum(last_error_but)/len(acc_each)])
        # print(one_correct_num+[sum(one_correct_num)/len(acc_each)])
        # print(one_error_num+[sum(one_error_num)/len(acc_each)])
    file_csv.close()


if __name__ == '__main__':
    main()
