from paddle import nn
from openrec.modeling.transforms import build_transform
from openrec.modeling.encoders import build_encoder
from openrec.modeling.decoders import build_decoder

__all__ = ["BaseModel"]


class BaseRecognizer(nn.Layer):
    def __init__(self, config):
        """
        the module for OCR.
        args:
            config (dict): the super parameters for module.
        """
        super(BaseRecognizer, self).__init__()
        in_channels = config.get("in_channels", 3)
        # build transfrom,
        # for rec, transfrom can be TPS,None
        if "Transform" not in config or config["Transform"] is None:
            self.use_transform = False
        else:
            self.use_transform = True
            config["Transform"]["in_channels"] = in_channels
            self.transform = build_transform(config["Transform"])
            in_channels = self.transform.out_channels

        # build backbone
        if "Encoder" not in config or config["Encoder"] is None:
            self.use_encoder = False
        else:
            self.use_encoder = True
            config["Encoder"]["in_channels"] = in_channels
            self.encoder = build_encoder(config["Encoder"])
            in_channels = self.encoder.out_channels

        # build head
        if "Decoder" not in config or config["Decoder"] is None:
            self.use_decoder = False
        else:
            self.use_decoder = True
            config["Decoder"]["in_channels"] = in_channels
            self.decoder = build_decoder(config["Decoder"])

        self.return_all_feats = config.get("return_all_feats", False)

    def forward(self, x, data=None):
        if self.use_transform:
            x = self.transform(x)
        if self.use_encoder:
            x = self.encoder(x)
        if self.use_decoder:
            x = self.decoder(x, data=data)
        return x
