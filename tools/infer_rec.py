import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))
import json
import numpy as np
import paddle

from openrec.preprocess import create_operators, transform
from openrec.modeling import build_model
from openrec.postprocess import build_post_process
from tools.utility import ArgsParser
from tools.engine import Config
from tools.utils.ckpt import load_ckpt
from tools.utils.utility import get_image_file_list
from tools.utils.logging import get_logger


def build_rec_process(cfg):
    transforms = []
    for op in cfg["Eval"]["dataset"]["transforms"]:
        op_name = list(op)[0]
        if "Label" in op_name:
            continue
        elif op_name in ["RecResizeImg"]:
            op[op_name]["infer_mode"] = True
        elif op_name == "KeepKeys":
            op[op_name]["keep_keys"] = ["image"]
        transforms.append(op)
    return transforms


def main(cfg):
    logger = get_logger()
    global_config = cfg["Global"]

    # build post process
    post_process_class = build_post_process(cfg["PostProcess"])

    char_num = len(getattr(post_process_class, "character"))
    cfg["Architecture"]["Decoder"]["out_channels"] = char_num
    model = build_model(cfg["Architecture"])
    load_ckpt(model, cfg)
    model.eval()

    # create data ops
    transforms = build_rec_process(cfg)
    global_config["infer_mode"] = True
    ops = create_operators(transforms, global_config)

    for file in get_image_file_list(global_config["infer_img"]):
        logger.info("infer_img: {}".format(file))
        with open(file, "rb") as f:
            img = f.read()
            data = {"image": img}
        batch = transform(data, ops)
        others = None
        images = np.expand_dims(batch[0], axis=0)
        images = paddle.to_tensor(images)
        with paddle.no_grad():
            preds = model(images, others)
        post_result = post_process_class(preds)
        if isinstance(post_result, dict):
            rec_info = dict()
            for key in post_result:
                if len(post_result[key][0]) >= 2:
                    rec_info[key] = {
                        "label": post_result[key][0][0],
                        "score": float(post_result[key][0][1]),
                    }
            info = json.dumps(rec_info, ensure_ascii=False)
        elif isinstance(post_result, list) and isinstance(post_result[0], int):
            # for RFLearning CNT branch
            info = str(post_result[0])
        else:
            if len(post_result[0]) >= 2:
                info = post_result[0][0] + "\t" + str(post_result[0][1])

        logger.info(f"{file}\t result: {info}")
    logger.info("success!")


if __name__ == "__main__":
    FLAGS = ArgsParser().parse_args()
    cfg = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop("opt")
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)
    main(cfg.cfg)
