import copy
from importlib import import_module

__all__ = ['build_post_process']

# 定义类名到模块路径的映射
module_mapping = {
    'CTCLabelDecode': '.ctc_postprocess',
    'CharLabelDecode': '.char_postprocess',
    'CELabelDecode': '.ce_postprocess',
    'CPPDLabelDecode': '.cppd_postprocess',
    'NRTRLabelDecode': '.nrtr_postprocess',
    'ABINetLabelDecode': '.abinet_postprocess',
    'ARLabelDecode': '.ar_postprocess',
    'IGTRLabelDecode': '.igtr_postprocess',
    'VisionLANLabelDecode': '.visionlan_postprocess',
    'SMTRLabelDecode': '.smtr_postprocess',
    'SRNLabelDecode': '.srn_postprocess',
    'LISTERLabelDecode': '.lister_postprocess',
    'MPGLabelDecode': '.mgp_postprocess',
    'GTCLabelDecode': '.'  # 当前模块中的类
}


def build_post_process(config, global_config=None):
    config = copy.deepcopy(config)
    module_name = config.pop('name')
    if global_config is not None:
        config.update(global_config)

    assert module_name in module_mapping, Exception(
        'post process only support {}'.format(list(module_mapping.keys())))

    module_path = module_mapping[module_name]

    # 处理当前模块中的类
    if module_path == '.':
        module_class = globals()[module_name]
    else:
        # 动态导入模块
        module = import_module(module_path, package=__package__)
        module_class = getattr(module, module_name)

    return module_class(**config)


class GTCLabelDecode(object):
    """Convert between text-label and text-index."""

    def __init__(self,
                 gtc_label_decode=None,
                 character_dict_path=None,
                 use_space_char=True,
                 only_gtc=False,
                 with_ratio=False,
                 **kwargs):
        gtc_label_decode['character_dict_path'] = character_dict_path
        gtc_label_decode['use_space_char'] = use_space_char
        self.gtc_label_decode = build_post_process(gtc_label_decode)
        self.ctc_label_decode = build_post_process({
            'name':
            'CTCLabelDecode',
            'character_dict_path':
            character_dict_path,
            'use_space_char':
            use_space_char
        })
        self.gtc_character = self.gtc_label_decode.character
        self.ctc_character = self.ctc_label_decode.character
        self.only_gtc = only_gtc
        self.with_ratio = with_ratio

    def get_character_num(self):
        return [len(self.gtc_character), len(self.ctc_character)]

    def __call__(self, preds, batch=None, *args, **kwargs):
        if self.with_ratio:
            batch = batch[:-1]
        gtc = self.gtc_label_decode(preds['gtc_pred'],
                                    batch[:-2] if batch is not None else None)
        if self.only_gtc:
            return gtc
        ctc = self.ctc_label_decode(preds['ctc_pred'], [None] +
                                    batch[-2:] if batch is not None else None)

        return [gtc, ctc]
