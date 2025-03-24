import argparse
import math

import cv2
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
import random


def str2bool(v):
    return v.lower() in ('true', 'yes', 't', 'y', '1')


def str2int_tuple(v):
    return tuple([int(i.strip()) for i in v.split(',')])


def init_args():
    parser = argparse.ArgumentParser()
    # params for prediction engine
    parser.add_argument('--use_gpu', type=str2bool, default=False)

    # params for text detector
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--det_algorithm', type=str, default='DB')
    parser.add_argument('--det_model_dir', type=str)
    parser.add_argument('--det_limit_side_len', type=float, default=960)
    parser.add_argument('--det_limit_type', type=str, default='max')
    parser.add_argument('--det_box_type', type=str, default='quad')

    # DB parmas
    parser.add_argument('--det_db_thresh', type=float, default=0.3)
    parser.add_argument('--det_db_box_thresh', type=float, default=0.6)
    parser.add_argument('--det_db_unclip_ratio', type=float, default=1.5)
    parser.add_argument('--max_batch_size', type=int, default=10)
    parser.add_argument('--use_dilation', type=str2bool, default=False)
    parser.add_argument('--det_db_score_mode', type=str, default='fast')

    # params for text recognizer
    parser.add_argument('--rec_algorithm', type=str, default='SVTR_LCNet')
    parser.add_argument('--rec_model_dir', type=str)
    parser.add_argument('--rec_image_inverse', type=str2bool, default=True)
    parser.add_argument('--rec_image_shape', type=str, default='3, 48, 320')
    parser.add_argument('--rec_batch_num', type=int, default=6)
    parser.add_argument('--max_text_length', type=int, default=25)
    parser.add_argument('--vis_font_path',
                        type=str,
                        default='./doc/fonts/simfang.ttf')
    parser.add_argument('--drop_score', type=float, default=0.5)

    # params for text classifier
    parser.add_argument('--use_angle_cls', type=str2bool, default=False)
    parser.add_argument('--cls_model_dir', type=str)
    parser.add_argument('--cls_image_shape', type=str, default='3, 48, 192')
    parser.add_argument('--label_list', type=list, default=['0', '180'])
    parser.add_argument('--cls_batch_num', type=int, default=6)
    parser.add_argument('--cls_thresh', type=float, default=0.9)

    parser.add_argument('--warmup', type=str2bool, default=False)

    #
    parser.add_argument('--output', type=str, default='./inference_results')
    parser.add_argument('--save_crop_res', type=str2bool, default=False)
    parser.add_argument('--crop_res_save_dir', type=str, default='./output')

    # multi-process
    parser.add_argument('--use_mp', type=str2bool, default=False)
    parser.add_argument('--total_process_num', type=int, default=1)
    parser.add_argument('--process_id', type=int, default=0)

    parser.add_argument('--show_log', type=str2bool, default=True)
    return parser


def parse_args():
    parser = init_args()
    return parser.parse_args()


def create_font(txt, sz, font_path='./doc/fonts/simfang.ttf'):
    font_size = int(sz[1] * 0.99)
    font = ImageFont.truetype(font_path, font_size, encoding='utf-8')
    if int(PIL.__version__.split('.')[0]) < 10:
        length = font.getsize(txt)[0]
    else:
        length = font.getlength(txt)

    if length > sz[0]:
        font_size = int(font_size * sz[0] / length)
        font = ImageFont.truetype(font_path, font_size, encoding='utf-8')
    return font


def draw_box_txt_fine(img_size, box, txt, font_path='./doc/fonts/simfang.ttf'):
    box_height = int(
        math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][1])**2))
    box_width = int(
        math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][1])**2))

    if box_height > 2 * box_width and box_height > 30:
        img_text = Image.new('RGB', (box_height, box_width), (255, 255, 255))
        draw_text = ImageDraw.Draw(img_text)
        if txt:
            font = create_font(txt, (box_height, box_width), font_path)
            draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)
        img_text = img_text.transpose(Image.ROTATE_270)
    else:
        img_text = Image.new('RGB', (box_width, box_height), (255, 255, 255))
        draw_text = ImageDraw.Draw(img_text)
        if txt:
            font = create_font(txt, (box_width, box_height), font_path)
            draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)

    pts1 = np.float32([[0, 0], [box_width, 0], [box_width, box_height],
                       [0, box_height]])
    pts2 = np.array(box, dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)

    img_text = np.array(img_text, dtype=np.uint8)
    img_right_text = cv2.warpPerspective(
        img_text,
        M,
        img_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return img_right_text


def draw_ocr_box_txt(
    image,
    boxes,
    txts=None,
    scores=None,
    drop_score=0.5,
    font_path='./doc/fonts/simfang.ttf',
):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = np.ones((h, w, 3), dtype=np.uint8) * 255
    random.seed(0)

    draw_left = ImageDraw.Draw(img_left)
    if txts is None or len(txts) != len(boxes):
        txts = [None] * len(boxes)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        if isinstance(box[0], list):
            box = list(map(tuple, box))
        draw_left.polygon(box, fill=color)
        img_right_text = draw_box_txt_fine((w, h), box, txt, font_path)
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_right_text, [pts], True, color, 1)
        img_right = cv2.bitwise_and(img_right, img_right_text)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(Image.fromarray(img_right), (w, 0, w * 2, h))
    return np.array(img_show)


def get_rotate_crop_image(img, points):
    """
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    """
    assert len(points) == 4, 'shape of points must be 4*2'
    img_crop_width = int(
        max(np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([
        [0, 0],
        [img_crop_width, 0],
        [img_crop_width, img_crop_height],
        [0, img_crop_height],
    ])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def get_minarea_rect_crop(img, points):
    bounding_box = cv2.minAreaRect(np.array(points).astype(np.int32))
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_a, index_b, index_c, index_d = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_a = 0
        index_d = 1
    else:
        index_a = 1
        index_d = 0
    if points[3][1] > points[2][1]:
        index_b = 2
        index_c = 3
    else:
        index_b = 3
        index_c = 2

    box = [points[index_a], points[index_b], points[index_c], points[index_d]]
    crop_img = get_rotate_crop_image(img, np.array(box))
    return crop_img


def check_gpu(use_gpu):
    import torch
    if use_gpu and not torch.cuda.is_available():
        use_gpu = False
    return use_gpu


if __name__ == '__main__':
    pass
