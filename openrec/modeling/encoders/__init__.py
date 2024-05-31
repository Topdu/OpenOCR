__all__ = ["build_encoder"]


def build_encoder(config):
    # from .rec_mobilenet_v3 import MobileNetV3
    from .resnet_vd import ResNet
    from .svtrnet import SVTRNet

    support_dict = [
        "MobileNetV1Enhance",
        "ResNet31",
        "MobileNetV3",
        "PPLCNetV3",
        "PPHGNet_small",
        "ResNet",
        "MTB",
        "SVTRNet",
    ]

    module_name = config.pop("name")
    assert module_name in support_dict, Exception(
        "when encoder of rec model only support {}".format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class
