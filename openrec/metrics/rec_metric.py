import string
import numpy as np
from rapidfuzz.distance import Levenshtein


def match_ss(ss1, ss2):
    s1_len = len(ss1)
    for c_i in range(s1_len):
        if ss1[c_i:] == ss2[:s1_len - c_i]:
            return ss2[s1_len - c_i:]
    return ss2


def stream_match(text):
    bs = len(text)
    s_list = []
    conf_list = []
    for s_conf in text:
        s_list.append(s_conf[0])
        conf_list.append(s_conf[1])
    s_n = bs
    s_start = s_list[0][:-1]
    s_new = s_start
    for s_i in range(1, s_n):
        s_start = match_ss(
            s_start, s_list[s_i][1:-1] if s_i < s_n - 1 else s_list[s_i][1:])
        s_new += s_start
    return s_new, sum(conf_list) / bs


class RecMetric(object):

    def __init__(self,
                 main_indicator='acc',
                 is_filter=False,
                 is_lower=True,
                 ignore_space=True,
                 stream=False,
                 with_ratio=False,
                 max_len=25,
                 max_ratio=4,
                 **kwargs):
        self.main_indicator = main_indicator
        self.is_filter = is_filter
        self.is_lower = is_lower
        self.ignore_space = ignore_space
        self.stream = stream
        self.eps = 1e-5
        self.with_ratio = with_ratio
        self.max_len = max_len
        self.max_ratio = max_ratio
        self.reset()

    def _normalize_text(self, text):
        text = ''.join(
            filter(lambda x: x in (string.digits + string.ascii_letters),
                   text))
        return text

    def __call__(self,
                 pred_label,
                 batch=None,
                 training=False,
                 *args,
                 **kwargs):
        if self.with_ratio and not training:
            return self.eval_all_metric(pred_label, batch)
        else:
            return self.eval_metric(pred_label)

    def eval_metric(self, pred_label, *args, **kwargs):
        preds, labels = pred_label
        correct_num = 0
        all_num = 0
        norm_edit_dis = 0.0
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
            if self.is_lower:
                pred = pred.lower()
                target = target.lower()
            norm_edit_dis += Levenshtein.normalized_distance(pred, target)
            if pred == target:
                correct_num += 1
            all_num += 1
        self.correct_num += correct_num
        self.all_num += all_num
        self.norm_edit_dis += norm_edit_dis
        return {
            'acc': correct_num / (all_num + self.eps),
            'norm_edit_dis': 1 - norm_edit_dis / (all_num + self.eps),
        }

    def eval_all_metric(self, pred_label, batch=None, *args, **kwargs):
        if self.with_ratio:
            ratio = batch[-1]
        preds, labels = pred_label
        correct_num = 0
        correct_num_real = 0
        correct_num_lower = 0
        correct_num_ignore_space = 0
        correct_num_ignore_space_lower = 0
        correct_num_ignore_space_symbol = 0
        all_num = 0
        norm_edit_dis = 0.0
        each_len_num = [0 for _ in range(self.max_len)]
        each_len_correct_num = [0 for _ in range(self.max_len)]
        each_len_norm_edit_dis = [0 for _ in range(self.max_len)]
        each_ratio_num = [0 for _ in range(self.max_ratio)]
        each_ratio_correct_num = [0 for _ in range(self.max_ratio)]
        each_ratio_norm_edit_dis = [0 for _ in range(self.max_ratio)]
        for (pred, pred_conf), (target, _) in zip(preds, labels):
            if self.stream:
                assert len(labels) == 1
                pred, _ = stream_match(preds)
            if pred == target:
                correct_num_real += 1

            if pred.lower() == target.lower():
                correct_num_lower += 1

            if self.ignore_space:
                pred = pred.replace(' ', '')
                target = target.replace(' ', '')
            if pred == target:
                correct_num_ignore_space += 1

            if pred.lower() == target.lower():
                correct_num_ignore_space_lower += 1

            if self.is_filter:
                pred = self._normalize_text(pred)
                target = self._normalize_text(target)
            if pred == target:
                correct_num_ignore_space_symbol += 1

            if self.is_lower:
                pred = pred.lower()
                target = target.lower()
            dis = Levenshtein.normalized_distance(pred, target)
            norm_edit_dis += dis
            ratio_i = ratio[all_num] - 1 if ratio[
                all_num] < self.max_ratio else self.max_ratio - 1
            len_i = max(0, min(self.max_len, len(target)) - 1)
            if pred == target:
                correct_num += 1
                each_len_correct_num[len_i] += 1
                each_ratio_correct_num[ratio_i] += 1
            each_len_num[len_i] += 1
            each_len_norm_edit_dis[len_i] += dis

            each_ratio_num[ratio_i] += 1
            each_ratio_norm_edit_dis[ratio_i] += dis
            all_num += 1
        self.correct_num += correct_num
        self.correct_num_real += correct_num_real
        self.correct_num_lower += correct_num_lower
        self.correct_num_ignore_space += correct_num_ignore_space
        self.correct_num_ignore_space_lower += correct_num_ignore_space_lower
        self.correct_num_ignore_space_symbol += correct_num_ignore_space_symbol
        self.all_num += all_num
        self.norm_edit_dis += norm_edit_dis
        self.each_len_num = self.each_len_num + np.array(each_len_num)
        self.each_len_correct_num = self.each_len_correct_num + np.array(
            each_len_correct_num)
        self.each_len_norm_edit_dis = self.each_len_norm_edit_dis + np.array(
            each_len_norm_edit_dis)
        self.each_ratio_num = self.each_ratio_num + np.array(each_ratio_num)
        self.each_ratio_correct_num = self.each_ratio_correct_num + np.array(
            each_ratio_correct_num)
        self.each_ratio_norm_edit_dis = self.each_ratio_norm_edit_dis + np.array(
            each_ratio_norm_edit_dis)
        return {
            'acc': correct_num / (all_num + self.eps),
            'norm_edit_dis': 1 - norm_edit_dis / (all_num + self.eps),
        }

    def get_metric(self, training=False):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        if self.with_ratio and not training:
            return self.get_all_metric()
        acc = 1.0 * self.correct_num / (self.all_num + self.eps)
        norm_edit_dis = 1 - self.norm_edit_dis / (self.all_num + self.eps)
        num_samples = self.all_num
        self.reset()
        return {
            'acc': acc,
            'norm_edit_dis': norm_edit_dis,
            'num_samples': num_samples
        }

    def get_all_metric(self):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        acc = 1.0 * self.correct_num / (self.all_num + self.eps)
        acc_real = 1.0 * self.correct_num_real / (self.all_num + self.eps)
        acc_lower = 1.0 * self.correct_num_lower / (self.all_num + self.eps)
        acc_ignore_space = 1.0 * self.correct_num_ignore_space / (
            self.all_num + self.eps)
        acc_ignore_space_lower = 1.0 * self.correct_num_ignore_space_lower / (
            self.all_num + self.eps)
        acc_ignore_space_symbol = 1.0 * self.correct_num_ignore_space_symbol / (
            self.all_num + self.eps)

        norm_edit_dis = 1 - self.norm_edit_dis / (self.all_num + self.eps)
        num_samples = self.all_num
        each_len_acc = (self.each_len_correct_num /
                        (self.each_len_num + self.eps)).tolist()
        each_len_norm_edit_dis = (1 -
                                  ((self.each_len_norm_edit_dis) /
                                   ((self.each_len_num) + self.eps))).tolist()
        each_len_num = self.each_len_num.tolist()
        each_ratio_acc = (self.each_ratio_correct_num /
                          (self.each_ratio_num + self.eps)).tolist()
        each_ratio_norm_edit_dis = (1 - ((self.each_ratio_norm_edit_dis) / (
            (self.each_ratio_num) + self.eps))).tolist()
        each_ratio_num = self.each_ratio_num.tolist()
        self.reset()
        return {
            'acc': acc,
            'acc_real': acc_real,
            'acc_lower': acc_lower,
            'acc_ignore_space': acc_ignore_space,
            'acc_ignore_space_lower': acc_ignore_space_lower,
            'acc_ignore_space_symbol': acc_ignore_space_symbol,
            'acc_ignore_space_lower_symbol': acc,
            'each_len_num': each_len_num,
            'each_len_acc': each_len_acc,
            'each_len_norm_edit_dis': each_len_norm_edit_dis,
            'each_ratio_num': each_ratio_num,
            'each_ratio_acc': each_ratio_acc,
            'each_ratio_norm_edit_dis': each_ratio_norm_edit_dis,
            'norm_edit_dis': norm_edit_dis,
            'num_samples': num_samples
        }

    def reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0
        self.correct_num_real = 0
        self.correct_num_lower = 0
        self.correct_num_ignore_space = 0
        self.correct_num_ignore_space_lower = 0
        self.correct_num_ignore_space_symbol = 0
        self.each_len_num = np.array([0 for _ in range(self.max_len)])
        self.each_len_correct_num = np.array([0 for _ in range(self.max_len)])
        self.each_len_norm_edit_dis = np.array(
            [0. for _ in range(self.max_len)])
        self.each_ratio_num = np.array([0 for _ in range(self.max_ratio)])
        self.each_ratio_correct_num = np.array(
            [0 for _ in range(self.max_ratio)])
        self.each_ratio_norm_edit_dis = np.array(
            [0. for _ in range(self.max_ratio)])
