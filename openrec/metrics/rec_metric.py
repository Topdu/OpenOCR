import string

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
                 ignore_space=True,
                 stream=False,
                 **kwargs):
        self.main_indicator = main_indicator
        self.is_filter = is_filter
        self.ignore_space = ignore_space
        self.stream = stream
        self.eps = 1e-5
        self.reset()

    def _normalize_text(self, text):
        text = ''.join(
            filter(lambda x: x in (string.digits + string.ascii_letters),
                   text))
        return text.lower()

    def __call__(self, pred_label, *args, **kwargs):
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

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        acc = 1.0 * self.correct_num / (self.all_num + self.eps)
        norm_edit_dis = 1 - self.norm_edit_dis / (self.all_num + self.eps)
        num_samples = self.all_num
        self.reset()
        return {
            'acc': acc,
            'norm_edit_dis': norm_edit_dis,
            'num_samples': num_samples
        }

    def reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0
