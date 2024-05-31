# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import paddle
from paddle.jit import to_static

from openrec.modeling import build_model
from openrec.postprocess import build_post_process
from tools.utils.save_load import load_model
from tools.utils.logging import get_logger
from tools.utility import ArgsParser
from tools.engine import Config

# from tools.program import load_config, merge_config, ArgsParser


def export_single_model(model,
                        arch_config,
                        save_path,
                        logger,
                        input_shape=None,
                        quanter=None):
    if arch_config["algorithm"] == "SRN":
        max_text_length = arch_config["Head"]["max_text_length"]
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 1, 64, 256], dtype="float32"), [
                    paddle.static.InputSpec(
                        shape=[None, 256, 1],
                        dtype="int64"), paddle.static.InputSpec(
                            shape=[None, max_text_length, 1], dtype="int64"),
                    paddle.static.InputSpec(
                        shape=[None, 8, max_text_length, max_text_length],
                        dtype="int64"), paddle.static.InputSpec(
                            shape=[None, 8, max_text_length, max_text_length],
                            dtype="int64")
                ]
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] == "SAR":
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 3, 48, 160], dtype="float32"),
            [paddle.static.InputSpec(
                shape=[None], dtype="float32")]
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] in ["SVTR_LCNet", "SVTR_HGNet"]:
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 3, 48, -1], dtype="float32"),
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] in ["SVTR", "CPPD"]:
        other_shape = [
            paddle.static.InputSpec(
                shape=[None] + input_shape, dtype="float32"),
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] == "PREN":
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 3, 64, 256], dtype="float32"),
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["model_type"] == "sr":
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 3, 16, 64], dtype="float32")
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] == "ViTSTR":
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 1, 224, 224], dtype="float32"),
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] == "ABINet":
        if not input_shape:
            input_shape = [3, 32, 128]
        other_shape = [
            paddle.static.InputSpec(
                shape=[None] + input_shape, dtype="float32"),
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] in ["NRTR", "SPIN", 'RFL']:
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 1, 32, 100], dtype="float32"),
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] in ['SATRN']:
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 3, 32, 100], dtype="float32"),
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] == "VisionLAN":
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 3, 64, 256], dtype="float32"),
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] == "RobustScanner":
        max_text_length = arch_config["Head"]["max_text_length"]
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 3, 48, 160], dtype="float32"), [
                    paddle.static.InputSpec(
                        shape=[None, ], dtype="float32"),
                    paddle.static.InputSpec(
                        shape=[None, max_text_length], dtype="int64")
                ]
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] == "CAN":
        other_shape = [[
            paddle.static.InputSpec(
                shape=[None, 1, None, None],
                dtype="float32"), paddle.static.InputSpec(
                    shape=[None, 1, None, None], dtype="float32"),
            paddle.static.InputSpec(
                shape=[None, arch_config['Head']['max_text_length']],
                dtype="int64")
        ]]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] in ["LayoutLM", "LayoutLMv2", "LayoutXLM"]:
        input_spec = [
            paddle.static.InputSpec(
                shape=[None, 512], dtype="int64"),  # input_ids
            paddle.static.InputSpec(
                shape=[None, 512, 4], dtype="int64"),  # bbox
            paddle.static.InputSpec(
                shape=[None, 512], dtype="int64"),  # attention_mask
            paddle.static.InputSpec(
                shape=[None, 512], dtype="int64"),  # token_type_ids
            paddle.static.InputSpec(
                shape=[None, 3, 224, 224], dtype="int64"),  # image
        ]
        if 'Re' in arch_config['Backbone']['name']:
            input_spec.extend([
                paddle.static.InputSpec(
                    shape=[None, 512, 3], dtype="int64"),  # entities
                paddle.static.InputSpec(
                    shape=[None, None, 2], dtype="int64"),  # relations
            ])
        if model.backbone.use_visual_backbone is False:
            input_spec.pop(4)
        model = to_static(model, input_spec=[input_spec])
    else:
        infer_shape = [3, -1, -1]
        if arch_config["model_type"] == "rec":
            infer_shape = [3, 32, -1]  # for rec model, H must be 32
            if "Transform" in arch_config and arch_config[
                    "Transform"] is not None and arch_config["Transform"][
                        "name"] == "TPS":
                logger.info(
                    "When there is tps in the network, variable length input is not supported, and the input size needs to be the same as during training"
                )
                infer_shape[-1] = 100
        elif arch_config["model_type"] == "table":
            infer_shape = [3, 488, 488]
            if arch_config["algorithm"] == "TableMaster":
                infer_shape = [3, 480, 480]
            if arch_config["algorithm"] == "SLANet":
                infer_shape = [3, -1, -1]
        model = to_static(
            model,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None] + infer_shape, dtype="float32")
            ])

    if arch_config["model_type"] != "sr" and arch_config["Backbone"][
            "name"] == "PPLCNetV3":
        # for rep lcnetv3
        for layer in model.sublayers():
            if hasattr(layer, "rep") and not getattr(layer, "is_repped"):
                layer.rep()

    if quanter is None:
        paddle.jit.save(model, save_path)
    else:
        quanter.save_quantized_model(model, save_path)
    logger.info("inference model is saved to {}".format(save_path))
    return


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--type", type=str, default="onnx", help="type of export")
    args = parser.parse_args()
    return args


def main():
    FLAGS = parse_args()
    config = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop("opt")
    config.merge_dict(FLAGS)
    config.merge_dict(opt)
    logger = get_logger()
    # build post process

    post_process_class = build_post_process(config["PostProcess"],
                                            config["Global"])

    # build model
    # for rec algorithm
    if hasattr(post_process_class, "character"):
        char_num = len(getattr(post_process_class, "character"))
        config["Architecture"]["Head"]["out_channels"] = char_num

    model = build_model(config["Architecture"])
    load_model(config, model, model_type=config['Architecture']["model_type"])
    model.eval()

    save_path = config["Global"]["save_inference_dir"]

    arch_config = config["Architecture"]

    if arch_config["algorithm"] in ["SVTR", "CPPD"] and arch_config["Head"][
            "name"] != 'MultiHead':
        input_shape = config["Eval"]["dataset"]["transforms"][-2]['SVTRResize'][
            'image_shape']
    elif arch_config["algorithm"].lower() == "ABINet".lower():
        rec_rs = [
            c for c in config["Eval"]["dataset"]["transforms"]
            if 'ABINetRecResizeImg' in c
        ]
        input_shape = rec_rs[0]['ABINetRecResizeImg'][
            'image_shape'] if rec_rs else None
    else:
        input_shape = None

    save_path = os.path.join(save_path, "inference")
    export_single_model(
        model, arch_config, save_path, logger, input_shape=input_shape)


if __name__ == "__main__":
    main()
