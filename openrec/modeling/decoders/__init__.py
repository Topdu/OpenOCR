__all__ = ["build_decoder"]


def build_decoder(config):
    # rec head
    from .ctc_decoder import CTCDecoder
    from .nrtr_decoder import NRTRDecoder
    from .cppd_decoder import CPPDDecoder

    support_dict = ["CTCDecoder", "NRTRDecoder", "CPPDDecoder"]

    module_name = config.pop("name")
    assert module_name in support_dict, Exception("head only support {}".format(
        support_dict))
    module_class = eval(module_name)(**config)
    return module_class
