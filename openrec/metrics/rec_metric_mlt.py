import numpy as np
from rapidfuzz.distance import Levenshtein
import unicodedata
import regex
import string

EPS = 1e-5


def _strip_spaces(s: str) -> str:
    if s is None:
        return ""
    return str(s).replace("\r", " ").replace("\n", " ").replace(" ", "").strip()


def _normalize_filter_symbols_like_metric(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s)
    s = regex.sub(r"\s+", "", s)
    s = regex.sub(r"[\p{P}\p{S}]+", "", s)
    return s


class RecMLTMetric(object):
    """
    对齐 MetricAgg 的指标口径：
      - acc_real: pred == gt
      - acc_lower: pred.lower() == gt.lower()
      - acc_ignore_space: strip_spaces(pred) == strip_spaces(gt)
      - acc_ignore_space_lower: strip_spaces(pred).lower() == strip_spaces(gt).lower()
      - acc_ignore_space_symbol: normalize_filter_symbols_like_metric(pred) == normalize_filter_symbols_like_metric(gt)
      - norm_edit_dis: 1 - mean(Levenshtein.normalized_distance(raw_pred, raw_gt))
      - acc: 使用 final_* 开关处理后的 pred/gt 再比较
    """

    def __init__(self,
                 main_indicator='acc',
                 with_ratio=False,
                 max_len=25,
                 max_ratio=4,
                 final_ignore_space=True,
                 final_is_filter=True,
                 final_is_lower=True,
                 **kwargs):
        self.main_indicator = main_indicator
        self.eps = EPS
        self.with_ratio = with_ratio
        self.max_len = max_len
        self.max_ratio = max_ratio

        # acc 的“最终比较口径”三开关（对齐 MetricAgg.add_one 的 final_*）
        self.final_ignore_space = final_ignore_space
        self.final_is_filter = final_is_filter
        self.final_is_lower = final_is_lower

        self.reset()

    def __call__(self, pred_label, batch=None, training=False, *args, **kwargs):
        if self.with_ratio and (not training):
            return self.eval_all_metric(pred_label, batch)
        else:
            return self.eval_metric(pred_label)

    def _finalize_pair(self, pred: str, gt: str):
        pred_final = pred
        gt_final = gt
        if self.final_ignore_space:
            pred_final = pred_final.replace(" ", "")
            gt_final = gt_final.replace(" ", "")
        if self.final_is_filter:
            pred_final = _normalize_filter_symbols_like_metric(pred_final)
            gt_final = _normalize_filter_symbols_like_metric(gt_final)
        if self.final_is_lower:
            pred_final = pred_final.lower()
            gt_final = gt_final.lower()
        return pred_final, gt_final

    def eval_metric(self, pred_label, *args, **kwargs):
        preds, labels = pred_label

        correct_num = 0
        correct_num_real = 0
        correct_num_lower = 0
        correct_num_ignore_space = 0
        correct_num_ignore_space_lower = 0
        correct_num_ignore_space_symbol = 0
        all_num = 0
        norm_edit_dis_sum = 0.0

        for (pred, _pred_conf), (gt, _gt_conf) in zip(preds, labels):
            pred = "" if pred is None else str(pred)
            gt = "" if gt is None else str(gt)

            # real / lower
            if pred == gt:
                correct_num_real += 1
            if pred.lower() == gt.lower():
                correct_num_lower += 1

            # ignore space
            pred_ns = _strip_spaces(pred)
            gt_ns = _strip_spaces(gt)
            if pred_ns == gt_ns:
                correct_num_ignore_space += 1
            if pred_ns.lower() == gt_ns.lower():
                correct_num_ignore_space_lower += 1

            # ignore space + symbols (按你 MetricAgg 的 normalize 口径)
            pred_sym = _normalize_filter_symbols_like_metric(pred)
            gt_sym = _normalize_filter_symbols_like_metric(gt)
            if pred_sym == gt_sym:
                correct_num_ignore_space_symbol += 1

            # norm edit distance: 用原始 pred/gt（不做 final 处理）
            dis = Levenshtein.normalized_distance(pred, gt)
            norm_edit_dis_sum += dis

            # acc: 用 final_* 处理后的 pred/gt
            pred_final, gt_final = self._finalize_pair(pred, gt)
            if pred_final == gt_final:
                correct_num += 1

            all_num += 1

        # 累加到全局
        self.correct_num += correct_num
        self.correct_num_real += correct_num_real
        self.correct_num_lower += correct_num_lower
        self.correct_num_ignore_space += correct_num_ignore_space
        self.correct_num_ignore_space_lower += correct_num_ignore_space_lower
        self.correct_num_ignore_space_symbol += correct_num_ignore_space_symbol
        self.all_num += all_num
        self.norm_edit_dis += norm_edit_dis_sum

        return {
            'acc': correct_num / (all_num + self.eps),
            'norm_edit_dis': 1.0 - norm_edit_dis_sum / (all_num + self.eps),
        }

    def eval_all_metric(self, pred_label, batch=None, *args, **kwargs):
        ratio = None
        if self.with_ratio and batch is not None:
            ratio = batch[-1]

        preds, labels = pred_label

        correct_num = 0
        correct_num_real = 0
        correct_num_lower = 0
        correct_num_ignore_space = 0
        correct_num_ignore_space_lower = 0
        correct_num_ignore_space_symbol = 0
        all_num = 0
        norm_edit_dis_sum = 0.0

        each_len_num = np.zeros(self.max_len, dtype=np.int64)
        each_len_correct_num = np.zeros(self.max_len, dtype=np.int64)
        each_len_norm_edit_dis = np.zeros(self.max_len, dtype=np.float64)

        each_ratio_num = np.zeros(self.max_ratio, dtype=np.int64)
        each_ratio_correct_num = np.zeros(self.max_ratio, dtype=np.int64)
        each_ratio_norm_edit_dis = np.zeros(self.max_ratio, dtype=np.float64)

        for idx, ((pred, _pred_conf), (gt, _gt_conf)) in enumerate(zip(preds, labels)):
            pred = "" if pred is None else str(pred)
            gt = "" if gt is None else str(gt)

            # 分桶索引
            if ratio is not None:
                r = ratio[idx]
                ratio_i = int(r - 1) if r < self.max_ratio else (self.max_ratio - 1)
            else:
                ratio_i = 0
            len_i = max(0, min(self.max_len, len(gt)) - 1)

            # 统计各类 acc（与 MetricAgg 一致）
            if pred == gt:
                correct_num_real += 1
            if pred.lower() == gt.lower():
                correct_num_lower += 1

            pred_ns = _strip_spaces(pred)
            gt_ns = _strip_spaces(gt)
            if pred_ns == gt_ns:
                correct_num_ignore_space += 1
            if pred_ns.lower() == gt_ns.lower():
                correct_num_ignore_space_lower += 1

            pred_sym = _normalize_filter_symbols_like_metric(pred)
            gt_sym = _normalize_filter_symbols_like_metric(gt)
            if pred_sym == gt_sym:
                correct_num_ignore_space_symbol += 1

            # norm edit distance: 原始 pred/gt
            dis = Levenshtein.normalized_distance(pred, gt)
            norm_edit_dis_sum += dis

            # acc: final 后比较（并用于分桶正确数）
            pred_final, gt_final = self._finalize_pair(pred, gt)
            is_correct = (pred_final == gt_final)
            if is_correct:
                correct_num += 1
                each_len_correct_num[len_i] += 1
                each_ratio_correct_num[ratio_i] += 1

            # 分桶计数/距离
            each_len_num[len_i] += 1
            each_len_norm_edit_dis[len_i] += dis
            each_ratio_num[ratio_i] += 1
            each_ratio_norm_edit_dis[ratio_i] += dis

            all_num += 1

        # 累加到全局
        self.correct_num += correct_num
        self.correct_num_real += correct_num_real
        self.correct_num_lower += correct_num_lower
        self.correct_num_ignore_space += correct_num_ignore_space
        self.correct_num_ignore_space_lower += correct_num_ignore_space_lower
        self.correct_num_ignore_space_symbol += correct_num_ignore_space_symbol
        self.all_num += all_num
        self.norm_edit_dis += norm_edit_dis_sum

        self.each_len_num += each_len_num
        self.each_len_correct_num += each_len_correct_num
        self.each_len_norm_edit_dis += each_len_norm_edit_dis

        self.each_ratio_num += each_ratio_num
        self.each_ratio_correct_num += each_ratio_correct_num
        self.each_ratio_norm_edit_dis += each_ratio_norm_edit_dis

        return {
            'acc': correct_num / (all_num + self.eps),
            'norm_edit_dis': 1.0 - norm_edit_dis_sum / (all_num + self.eps),
        }

    def get_metric(self):
        n = self.all_num

        acc = self.correct_num / (n + self.eps)
        acc_real = self.correct_num_real / (n + self.eps)
        acc_lower = self.correct_num_lower / (n + self.eps)
        acc_ignore_space = self.correct_num_ignore_space / (n + self.eps)
        acc_ignore_space_lower = self.correct_num_ignore_space_lower / (n + self.eps)
        acc_ignore_space_symbol = self.correct_num_ignore_space_symbol / (n + self.eps)
        norm_edit_dis = 1.0 - (self.norm_edit_dis / (n + self.eps))

        each_len_acc = (self.each_len_correct_num / (self.each_len_num + self.eps)).tolist()
        each_len_norm_edit_dis = (1.0 - (self.each_len_norm_edit_dis / (self.each_len_num + self.eps))).tolist()
        each_len_num = self.each_len_num.tolist()

        each_ratio_acc = (self.each_ratio_correct_num / (self.each_ratio_num + self.eps)).tolist()
        each_ratio_norm_edit_dis = (1.0 - (self.each_ratio_norm_edit_dis / (self.each_ratio_num + self.eps))).tolist()
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
            'num_samples': n,
        }

    def reset(self):
        self.correct_num = 0
        self.correct_num_real = 0
        self.correct_num_lower = 0
        self.correct_num_ignore_space = 0
        self.correct_num_ignore_space_lower = 0
        self.correct_num_ignore_space_symbol = 0

        self.all_num = 0
        self.norm_edit_dis = 0.0

        self.each_len_num = np.zeros(self.max_len, dtype=np.int64)
        self.each_len_correct_num = np.zeros(self.max_len, dtype=np.int64)
        self.each_len_norm_edit_dis = np.zeros(self.max_len, dtype=np.float64)

        self.each_ratio_num = np.zeros(self.max_ratio, dtype=np.int64)
        self.each_ratio_correct_num = np.zeros(self.max_ratio, dtype=np.int64)
        self.each_ratio_norm_edit_dis = np.zeros(self.max_ratio, dtype=np.float64)
