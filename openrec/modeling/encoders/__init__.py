__all__ = ['build_encoder']


def build_encoder(config):
    # from .rec_mobilenet_v3 import MobileNetV3
    from .focalsvtr import FocalSVTR
    from .rec_hgnet import PPHGNet_small
    from .rec_lcnetv3 import PPLCNetV3
    from .rec_mv1_enhance import MobileNetV1Enhance
    from .rec_nrtr_mtb import MTB
    from .rec_resnet_31 import ResNet31
    from .rec_resnet_45 import ResNet45
    from .rec_resnet_fpn import ResNet_FPN
    from .rec_resnet_vd import ResNet
    from .resnet31_rnn import ResNet_ASTER
    from .svtrnet import SVTRNet
    from .svtrnet2dpos import SVTRNet2DPos
    from .svtrv2 import SVTRv2
    from .svtrv2_lnconv import SVTRv2LNConv
    from .svtrv2_lnconv_two33 import SVTRv2LNConvTwo33
    from .vit import ViT
    support_dict = [
        'MobileNetV1Enhance', 'ResNet31', 'MobileNetV3', 'PPLCNetV3',
        'PPHGNet_small', 'ResNet', 'MTB', 'SVTRNet', 'ResNet45', 'ViT',
        'SVTRNet2DPos', 'SVTRv2', 'FocalSVTR', 'ResNet_FPN', 'ResNet_ASTER',
        'SVTRv2LNConv', 'SVTRv2LNConvTwo33'
    ]

    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        'when encoder of rec model only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class
