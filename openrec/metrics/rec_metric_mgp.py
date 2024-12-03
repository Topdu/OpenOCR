from .rec_metric import RecMetric


class RecMPGMetric(object):

    def __init__(self,
                 main_indicator='acc',
                 is_filter=False,
                 ignore_space=True,
                 stream=False,
                 with_ratio=False,
                 max_len=25,
                 max_ratio=4,
                 **kwargs):
        self.main_indicator = main_indicator
        self.is_filter = is_filter
        self.ignore_space = ignore_space
        self.eps = 1e-5
        self.char_metric = RecMetric(main_indicator=main_indicator,
                                     is_filter=is_filter,
                                     ignore_space=ignore_space,
                                     stream=stream,
                                     with_ratio=with_ratio,
                                     max_len=max_len,
                                     max_ratio=max_ratio)
        self.bpe_metric = RecMetric(main_indicator=main_indicator,
                                    is_filter=is_filter,
                                    ignore_space=ignore_space,
                                    stream=stream,
                                    with_ratio=with_ratio,
                                    max_len=max_len,
                                    max_ratio=max_ratio)

        self.wp_metric = RecMetric(main_indicator=main_indicator,
                                   is_filter=is_filter,
                                   ignore_space=ignore_space,
                                   stream=stream,
                                   with_ratio=with_ratio,
                                   max_len=max_len,
                                   max_ratio=max_ratio)
        self.final_metric = RecMetric(main_indicator=main_indicator,
                                      is_filter=is_filter,
                                      ignore_space=ignore_space,
                                      stream=stream,
                                      with_ratio=with_ratio,
                                      max_len=max_len,
                                      max_ratio=max_ratio)

    def __call__(self,
                 pred_label,
                 batch=None,
                 training=False,
                 *args,
                 **kwargs):

        char_metric = self.char_metric((pred_label[0], pred_label[-1]),
                                       batch,
                                       training=training)
        bpe_metric = self.bpe_metric((pred_label[1], pred_label[-1]),
                                     batch,
                                     training=training)
        wp_metric = self.wp_metric((pred_label[2], pred_label[-1]),
                                   batch,
                                   training=training)
        final_metric = self.final_metric((pred_label[3], pred_label[-1]),
                                         batch,
                                         training=training)
        final_metric['char_acc'] = char_metric['acc']
        final_metric['char_norm_edit_dis'] = char_metric['norm_edit_dis']
        final_metric['bpe_acc'] = bpe_metric['acc']
        final_metric['bpe_norm_edit_dis'] = bpe_metric['norm_edit_dis']
        final_metric['wp_acc'] = wp_metric['acc']
        final_metric['wp_norm_edit_dis'] = wp_metric['norm_edit_dis']
        return final_metric

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        char_metric = self.char_metric.get_metric()
        bpe_metric = self.bpe_metric.get_metric()
        wp_metric = self.wp_metric.get_metric()
        final_metric = self.final_metric.get_metric()
        final_metric['char_acc'] = char_metric['acc']
        final_metric['char_norm_edit_dis'] = char_metric['norm_edit_dis']
        final_metric['bpe_acc'] = bpe_metric['acc']
        final_metric['bpe_norm_edit_dis'] = bpe_metric['norm_edit_dis']
        final_metric['wp_acc'] = wp_metric['acc']
        final_metric['wp_norm_edit_dis'] = wp_metric['norm_edit_dis']
        return final_metric
