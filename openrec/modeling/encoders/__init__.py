__all__ = ['build_encoder']

from importlib import import_module

name_to_module = {
    'MobileNetV1Enhance': '.rec_mv1_enhance',
    'ResNet31': '.rec_resnet_31',
    'MobileNetV3': '.rec_mobilenet_v3',
    'PPLCNetV3': '.rec_lcnetv3',
    'PPHGNet_small': '.rec_hgnet',
    'ResNet': '.rec_resnet_vd',
    'MTB': '.rec_nrtr_mtb',
    'SVTRNet': '.svtrnet',
    'ResNet45': '.rec_resnet_45',
    'ViT': '.vit',
    'SVTRNet2DPos': '.svtrnet2dpos',
    'SVTRv2': '.svtrv2',
    'FocalSVTR': '.focalsvtr',
    'ResNet_FPN': '.rec_resnet_fpn',
    'ResNet_ASTER': '.resnet31_rnn',
    'SVTRv2LNConv': '.svtrv2_lnconv',
    'SVTRv2LNConvTwo33': '.svtrv2_lnconv_two33',
    'CAMEncoder': '.cam_encoder',
    'ConvNeXtV2': '.convnextv2',
    'AutoSTREncoder': '.autostr_encoder',
    'NRTREncoder': '.nrtr_encoder',
    'RepSVTREncoder': '.repvit',
}


def build_encoder(config):

    module_name = config.pop('name')
    assert module_name in name_to_module, Exception(
        f'Encoder only supports: {list(name_to_module.keys())}')

    module_path = name_to_module[module_name]
    mod = import_module(module_path, package=__package__)
    module_class = getattr(mod, module_name)(**config)

    return module_class
