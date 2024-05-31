import string

import numpy as np
from rapidfuzz.distance import Levenshtein

from .rec_metric import stream_match

# f_pred = open('pred_focal_subs_rand1_h2_bi_first.txt', 'w')


class RecMetricLong(object):

    def __init__(self,
                 main_indicator='acc',
                 is_filter=False,
                 ignore_space=True,
                 stream=False,
                 **kwargs):
        self.main_indicator = main_indicator
        self.is_filter = is_filter
        self.ignore_space = ignore_space
        self.stream = stream
        self.eps = 1e-5
        self.max_len = 201
        self.reset()

    def _normalize_text(self, text):
        text = ''.join(
            filter(lambda x: x in (string.digits + string.ascii_letters),
                   text))
        return text.lower()

    def __call__(self, pred_label, *args, **kwargs):
        preds, labels = pred_label
        correct_num = 0
        correct_num_slice = 0
        f_l_acc = 0
        all_num = 0
        norm_edit_dis = 0.0
        len_acc = 0
        each_len_num = [0 for _ in range(self.max_len)]
        each_len_correct_num = [0 for _ in range(self.max_len)]
        each_len_norm_edit_dis = [0 for _ in range(self.max_len)]
        for (pred, pred_conf), (target, _) in zip(preds, labels):
            if self.stream:
                assert len(labels) == 1
                pred, _ = stream_match(preds)
            if self.ignore_space:
                pred = pred.replace(' ', '')
                target = target.replace(' ', '')
            if self.is_filter:
                pred = self._normalize_text(pred)
                target = self._normalize_text(target)
            dis = Levenshtein.normalized_distance(pred, target)
            norm_edit_dis += dis
            # print(pred, target)
            if pred == target:
                correct_num += 1
                each_len_correct_num[len(target)] += 1
            each_len_num[len(target)] += 1
            each_len_norm_edit_dis[len(target)] += dis
            #     f_pred.write(pred+'\t'+target+'\t1'+'\n')
            #     print(pred, target, 1)
            # else:
            #     f_pred.write(pred+'\t'+target+'\t0'+'\n')
            #     print(pred, target, 0)
            if len(pred) >= 1 and len(target) >= 1:
                if pred[0] == target[0] and pred[-1] == target[-1]:
                    f_l_acc += 1
            if len(pred) == len(target):
                len_acc += 1
            if pred == target[:len(pred)]:
                # if pred == target[-len(pred):]:
                correct_num_slice += 1
            all_num += 1
        self.correct_num += correct_num
        self.correct_num_slice += correct_num_slice
        self.f_l_acc += f_l_acc
        self.all_num += all_num
        self.len_acc += len_acc
        self.each_len_num = self.each_len_num + np.array(each_len_num)
        self.each_len_correct_num = self.each_len_correct_num + np.array(
            each_len_correct_num)
        self.each_len_norm_edit_dis = self.each_len_norm_edit_dis + np.array(
            each_len_norm_edit_dis)
        self.norm_edit_dis += norm_edit_dis
        return {
            'acc': correct_num / (all_num + self.eps),
            'norm_edit_dis': 1 - norm_edit_dis / (all_num + self.eps),
        }

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        acc = 1.0 * self.correct_num / (self.all_num + self.eps)
        acc_slice = 1.0 * self.correct_num_slice / (self.all_num + self.eps)
        f_l_acc = 1.0 * self.f_l_acc / (self.all_num + self.eps)
        len_acc = 1.0 * self.len_acc / (self.all_num + self.eps)
        norm_edit_dis = 1 - self.norm_edit_dis / (self.all_num + self.eps)
        each_len_acc = (self.each_len_correct_num /
                        (self.each_len_num + self.eps)).tolist()
        # each_len_acc_25 = each_len_acc[:26]
        # each_len_acc_26 = each_len_acc[26:]
        each_len_norm_edit_dis = (1 -
                                  ((self.each_len_norm_edit_dis) /
                                   ((self.each_len_num) + self.eps))).tolist()
        # each_len_norm_edit_dis_25 = each_len_norm_edit_dis[:26]
        # each_len_norm_edit_dis_26 = each_len_norm_edit_dis[26:]
        each_len_num = self.each_len_num.tolist()
        all_num = self.all_num
        self.reset()
        return {
            'acc': acc,
            'norm_edit_dis': norm_edit_dis,
            'acc_slice': acc_slice,
            'f_l_acc': f_l_acc,
            'len_acc': len_acc,
            'each_len_num': each_len_num,
            'each_len_acc': each_len_acc,
            # "each_len_acc_25": each_len_acc_25,
            # "each_len_acc_26": each_len_acc_26,
            'each_len_norm_edit_dis': each_len_norm_edit_dis,
            # "each_len_norm_edit_dis_25":each_len_norm_edit_dis_25,
            # "each_len_norm_edit_dis_26":each_len_norm_edit_dis_26,
            'all_num': all_num
        }

    def reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0
        self.correct_num_slice = 0
        self.each_len_num = np.array([0 for _ in range(self.max_len)])
        self.each_len_correct_num = np.array([0 for _ in range(self.max_len)])
        self.each_len_norm_edit_dis = np.array(
            [0. for _ in range(self.max_len)])
        self.f_l_acc = 0
        self.len_acc = 0
