import io
import json
import re
import os
import random
import traceback
import ast
from PIL import Image
import fitz
import lmdb
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from openrec.preprocess import create_operators, transform
from openrec.preprocess.rec_aug import DocAug  #, PARSeqAugPIL
from pdf2image import convert_from_path

import torch.distributed as dist

import unicodedata

# 完整音标表
COMBINING_DIACRITICS = {
    '´': '\u0301',
    '`': '\u0300',
    'ˆ': '\u0302',
    '¨': '\u0308',
    'ˇ': '\u030C',
    '˘': '\u0306',
    '¯': '\u0304',
    '˚': '\u030A',
    '˜': '\u0303',
    '˙': '\u0307',
    '¸': '\u0327',
    '˛': '\u0328',
    '̇': '\u0307',
    '̱': '\u0331',
    '̥': '\u0325',
    '̩': '\u0329',
    '̯': '\u032F',
    '̄': '\u0304',
    '̋': '\u030B',
    '̨': '\u0328',
}

# 正则表达式匹配 “音标 + 字符”
pattern = re.compile(r'([' + re.escape(''.join(COMBINING_DIACRITICS.keys())) +
                     r'])([A-Za-z])')


def fix_diacritics_regex(text):

    def repl(match):
        accent = COMBINING_DIACRITICS[match.group(1)]
        base = match.group(2)
        return unicodedata.normalize('NFC', base + accent)

    return pattern.sub(repl, text)


def random_color():
    """生成一个随机的 RGB 颜色元组 (0.0 到 1.0 之间)."""
    return (random.random(), random.random(), random.random())


def load_pdf_as_image(pdf_path, dpi=300):

    images = convert_from_path(pdf_path, dpi=dpi)
    return images[0]


def get_masked_image(
        image,
        page_info,
        box_padding=3,
        mask_color=(255, 255, 255),
        token_num_limit=(15, 5000),
        overlap_ratio_limit=0.003,
):
    # box_padding = max(0, int(DPI * box_padding_ratio) )

    tot_num = sum([block['token_num'] for block in page_info])
    if tot_num < token_num_limit[0] or tot_num > token_num_limit[1]:
        return None

    box_list = [
        block.get('vision_content', {}).get('box_list', '[]')
        for block in page_info
    ]
    box_list = [ast.literal_eval(box) for box in box_list]
    box_list = [item for lst in box_list for item in lst]
    # box = (top, bottom, left, right)

    tex_token_info_list = [
        block.get('vision_content', {}).get('tex_token_info_list', [])
        for block in page_info
    ]
    tex_token_info_list = [
        ast.literal_eval(info) for info in tex_token_info_list
    ]
    tex_token_info_list = [item for lst in tex_token_info_list for item in lst]
    # tex_token_info = ("chars", "pos", "len", "attr")

    assert len(box_list) == len(tex_token_info_list)

    width, height = image.size
    mask_matrix = np.zeros((height, width), dtype=np.uint8)
    # for box in box_list:
    additonal_mask_matrix = np.zeros((height, width), dtype=np.uint8)
    for tex_token_info, box in zip(tex_token_info_list, box_list):
        if box == [0, 0, 0, 0]:
            continue
        top, bottom, left, right = box
        top, bottom, left, right = max(0, top - box_padding), min(
            height - 1, bottom + box_padding), max(0, left - box_padding), min(
                width - 1, right + box_padding)
        # mask_matrix[top:bottom+1, left:right+1] += 1
        if tex_token_info[3] not in ('tabular', ):
            mask_matrix[top:bottom + 1, left:right + 1] += 1
        else:
            additonal_mask_matrix[top:bottom + 1, left:right + 1] += 1

    if mask_matrix.max() == 0:
        return None
    overlap_matrix = (mask_matrix > 1)
    overlap_ratio = overlap_matrix.sum() / (width * height)
    if overlap_ratio > overlap_ratio_limit:
        return None
    # print(f"overlap ratio: {overlap_ratio}")

    mask_matrix = mask_matrix + additonal_mask_matrix
    mask_matrix = (mask_matrix == 0)
    img_array = np.array(image)
    col = np.array(mask_color, img_array.dtype)
    img_array[mask_matrix] = col
    masked_image = Image.fromarray(img_array)

    return masked_image


layout_map = {
    'section_paragraph': 'paragraph',
    'subsection_paragraph': 'paragraph',
    'subsubsection_paragraph': 'paragraph',
    'document_paragraph': 'paragraph',
    'enumerate_paragraph': 'paragraph',
    'enumerate*_paragraph': 'paragraph',
    'itemize_paragraph': 'paragraph',
    'itemize*_paragraph': 'paragraph',
    'paragraph_paragraph': 'paragraph',
    'subparagraph_paragraph': 'paragraph',
    'chapter_paragraph': 'paragraph',
    'part_paragraph': 'paragraph',
    'figure_paragraph': 'figure',
    'figure*_paragraph': 'figure',
    'caption_paragraph': 'caption',
    'caption*_paragraph': 'caption',
    'table_paragraph': 'table',
    'table*_paragraph': 'table',
    'deluxetable_paragraph': 'table',
    'deluxetable*_paragraph': 'table',
    'section_title': 'section_title',
    'subsection_title': 'subsection_title',
    'subsubsection_title': 'subsubsection_title',
    'subparagraph_title': 'subsection_title',
    'title_paragraph': 'title',
    'abstract_paragraph': 'abstract',
    'abstract*_paragraph': 'abstract',
    'footnote_paragraph': 'footnote',
    'tablenotes_paragraph': 'tablenotes',
    # footer
}

pattern_indent = re.compile(r'\\,|\\;|\\:|\\!|\\\s+')


def rm_indent_in_latex(text):
    text = pattern_indent.sub('', text)
    return text


def resize_image(original_width, original_height, max_width, max_height):
    # 计算宽高比
    aspect_ratio = original_width / original_height

    # 计算新的宽度和高度
    if original_width > max_width or original_height > max_height:
        if (max_width / max_height) >= aspect_ratio:
            # 按高度限制比例
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
        else:
            # 按宽度限制比例
            new_width = max_width
            new_height = int(new_width / aspect_ratio)
    else:
        # 如果图片已经小于或等于最大尺寸，则无需调整
        new_width, new_height = original_width, original_height
    return new_width, new_height


class NaSizeDataSet(Dataset):

    def __init__(self, config, mode, logger, seed=None, epoch=0, task='Rec'):
        super(NaSizeDataSet, self).__init__()
        self.logger = logger
        self.mode = mode.lower()

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0
        num_replicas = world_size

        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']
        self.seed = seed if seed is not None else epoch
        random.seed(self.seed)
        self.e2e_info = dataset_config.get('e2e_info', True)
        self.layout_info = dataset_config.get('layout_info', False)
        self.add_return = dataset_config.get('add_return', True)
        self.zoom_min_factor = dataset_config.get('zoom_min_factor', 10)
        self.use_zoom = dataset_config.get('use_zoom', False)
        self.all_data = dataset_config.get('all_data', False)
        self.use_linedata = dataset_config.get('use_linedata', False)
        self.test_data = dataset_config.get('test_data', False)
        self.use_aug = dataset_config.get('use_aug', True)
        self.use_table = dataset_config.get('use_table', False)
        self.e2e_info = False if self.layout_info else self.e2e_info

        self.use_math_norm = dataset_config.get('use_math_norm', False)
        self.root_path = dataset_config.get('root_path', None)
        if self.root_path is None:
            assert False, 'root_path is None'
        self.lmdb_path = dataset_config.get(
            'lmdb_path', f'{self.root_path}/output_pdf_lmdb/pdf_lmdb')
        self.env = None  # LMDB environment
        img_label_pair_list = {}
        self.do_shuffle = loader_config['shuffle']

        self.max_side = dataset_config.get('max_side',
                                           [64 * 15, 64 * 22])  # w, h
        self.divided_factor = dataset_config.get('divided_factor',
                                                 [64., 64.])  # w, h
        self.use_region = dataset_config.get('use_region', False)
        self.use_ch = dataset_config.get('use_ch', False)
        logger.info('Initialize indexs of doc datasets')
        list_file_json = []
        if self.all_data:
            self.__init_lmdb()

        if self.test_data:
            epoch_current = (epoch - 1) % 4
            region_4ep_ch = f'{self.root_path}/ch_pdf_label/40w_e2e_test1_croppdf_ch/region_4ep/{epoch_current}'
            for file_na in os.listdir(region_4ep_ch):
                if file_na.endswith('.json'):
                    list_file_json.append(os.path.join(region_4ep_ch, file_na))
        else:
            label_path_block_pymu = f'{self.root_path}/40w_e2e_test1_copy/block_pymu_fix_none_line/'
            for file_na in os.listdir(label_path_block_pymu):
                if file_na.endswith('.json'):
                    list_file_json.append(
                        os.path.join(label_path_block_pymu, file_na))
            epoch_current = (epoch - 1) % 10
            block_pymu_case_title_10ep = f'{self.root_path}/40w_e2e_test1_copy/block_pymu_case_fix_none_line_10ep/{epoch_current}'
            for file_na in os.listdir(block_pymu_case_title_10ep):
                if file_na.endswith('.json'):
                    list_file_json.append(
                        os.path.join(block_pymu_case_title_10ep, file_na))
            epoch_current = (epoch - 1) % 20
            rec_label_all_region_10ep = f'{self.root_path}/40w_e2e_test1_copy/rec_label_all_region_fix_none_line_20ep/{epoch_current}'
            for file_na in os.listdir(rec_label_all_region_10ep):
                if file_na.endswith('.json'):
                    list_file_json.append(
                        os.path.join(rec_label_all_region_10ep, file_na))
            epoch_current = (epoch - 1) % 5
            math_all_5ep = f'{self.root_path}/40w_e2e_test1_copy/math_all_5ep/{epoch_current}'
            for file_na in os.listdir(math_all_5ep):
                if file_na.endswith('.json'):
                    list_file_json.append(os.path.join(math_all_5ep, file_na))

            if self.use_ch:
                math_inline_ch = f'{self.root_path}/ch_pdf_label/40w_e2e_test1_croppdf_ch/math_inline/'
                for file_na in os.listdir(math_inline_ch):
                    if file_na.endswith('.json'):
                        list_file_json.append(
                            os.path.join(math_inline_ch, file_na))

                math_display_ch = f'{self.root_path}/ch_pdf_label/40w_e2e_test1_croppdf_ch/math_display/'
                for file_na in os.listdir(math_display_ch):
                    if file_na.endswith('.json'):
                        list_file_json.append(
                            os.path.join(math_display_ch, file_na))
                epoch_current = (epoch - 1) % 4
                plain_text_4ep_ch = f'{self.root_path}/ch_pdf_label/40w_e2e_test1_croppdf_ch/plain_text_4ep/{epoch_current}'
                for file_na in os.listdir(plain_text_4ep_ch):
                    if file_na.endswith('.json'):
                        list_file_json.append(
                            os.path.join(plain_text_4ep_ch, file_na))

                region_4ep_ch = f'{self.root_path}/ch_pdf_label/40w_e2e_test1_croppdf_ch/region_4ep/{epoch_current}'
                for file_na in os.listdir(region_4ep_ch):
                    if file_na.endswith('.json'):
                        list_file_json.append(
                            os.path.join(region_4ep_ch, file_na))

                math_inline_ch = f'{self.root_path}/ch_pdf_label/40w_e2e_test1_croppdf_ch_random_new/math_inline/'
                for file_na in os.listdir(math_inline_ch):
                    if file_na.endswith('.json'):
                        list_file_json.append(
                            os.path.join(math_inline_ch, file_na))

                math_display_ch = f'{self.root_path}/ch_pdf_label/40w_e2e_test1_croppdf_ch_random_new/math_display/'
                for file_na in os.listdir(math_display_ch):
                    if file_na.endswith('.json'):
                        list_file_json.append(
                            os.path.join(math_display_ch, file_na))
                epoch_current = (epoch - 1) % 4
                plain_text_4ep_ch = f'{self.root_path}/ch_pdf_label/40w_e2e_test1_croppdf_ch_random_new/plain_text_4ep/{epoch_current}'
                for file_na in os.listdir(plain_text_4ep_ch):
                    if file_na.endswith('.json'):
                        list_file_json.append(
                            os.path.join(plain_text_4ep_ch, file_na))

                region_4ep_ch = f'{self.root_path}/ch_pdf_label/40w_e2e_test1_croppdf_ch_random_new/region_4ep/{epoch_current}'
                for file_na in os.listdir(region_4ep_ch):
                    if file_na.endswith('.json'):
                        list_file_json.append(
                            os.path.join(region_4ep_ch, file_na))

        for file_json in list_file_json:
            w_r, h_r = os.path.basename(file_json)[:-5].split('_')[2:]
            w_r = int(w_r)
            h_r = int(h_r)
            if w_r < self.divided_factor[1] or h_r < self.divided_factor[0]:
                continue
            try:
                with open(file_json, 'r') as f:
                    # print(f'Loading {file_json}...')
                    json_data_list = json.load(f)
            except:
                print(f'Loading {file_json} ... error')
                continue
            w_r, h_r = resize_image(w_r, h_r, self.max_side[0],
                                    self.max_side[1])
            h_r = max(
                int(h_r // self.divided_factor[1] * self.divided_factor[1]),
                self.divided_factor[1])
            w_r = max(
                int(w_r // self.divided_factor[0] * self.divided_factor[0]),
                self.divided_factor[0])

            key = str(w_r) + '_' + str(h_r)
            if key in img_label_pair_list:
                img_label_pair_list[key].extend(json_data_list)
            else:
                img_label_pair_list[key] = json_data_list
        self.docaug = DocAug()
        if self.use_linedata and self.all_data:
            epoch_current = (epoch - 1) % 10
            line_json_list = [
                f'{self.root_path}/K-12/label_key_qwen.json',  # 25w tex_norm 1
                f'{self.root_path}/LSVT-2019/label_key.json',  # 260868 5
                f'{self.root_path}/webdata_MTWI/label_key.json',  # 147160 7
                f'{self.root_path}/HWDB2Train/label_key.json',  # 376029 1
                f'{self.root_path}/HWDB2Train/label_key_region.json',  # 36684 10
                f'{self.root_path}/TAL_OCR_HW/label_key.json',  # 2w 10
                f'{self.root_path}/K-12_exam/label_key_qwen.json',  # 5w tex_norm 3
                f'{self.root_path}/hw_pdf/label_key_qwen_tex_norm.json',  # 5.5k 5368 10
                f'{self.root_path}/hw_pdf/label_key_qwen_crop_tex_norm.json',  # 8.1w 83723 83755 5
                f'{self.root_path}/hw_xhs/label_key_qwen_tex_norm.json',  # 73826 73911 5
                f'{self.root_path}/dfcf_pdf_dpi300/label_key.json',  # 349907 fix huice dpi300 3
                f'{self.root_path}/huaxue_jiaoyu/label_key.json',  # 263000 # formula 2
                f'{self.root_path}/latex_aug_new_circled/label_key.json',  # 215000 223000 # rm only textcircled  2
                f'{self.root_path}/nongminribao_pdf_dpi300/label_keys.json',  #126963 dpi300 2
                f'{self.root_path}/hiertext_lmdb/label_key_char_line_para.json',  # 25w 1
                f'{self.root_path}/hiertext_lmdb/label_key_word_{epoch_current}ep.json',  # 8w 1
            ]
            ratio_sample = [1, 5, 7, 1, 10, 10, 3, 10, 5, 5, 3, 5, 5, 2, 1,
                            1]  # ratio_sample for line_json_list
            if self.use_table:
                line_json_list += [
                    f'{self.root_path}/hw_pdf_table/label_key.json',  #677
                    f'{self.root_path}/dfcf_table/label_key_refine.json',  # 218330
                    f'{self.root_path}/jiaoyu_table/label_key.json',  # 3233
                    f'{self.root_path}/pubtab1m_table/label_key_refine.json',  # 131463
                ]
                ratio_sample += [50, 3, 50, 3]  # for table

            self.__init_line_lmdb(epoch_current)
            for line_json, rti_sam in zip(line_json_list, ratio_sample):
                with open(line_json, 'r') as f:
                    json_data_list = json.load(f)
                    for keywh, value_list in json_data_list.items():
                        if rti_sam > 1:
                            value_list = value_list * rti_sam
                        elif rti_sam < 1:
                            value_list = value_list[:int(
                                len(value_list) * rti_sam)]

                        w_r, h_r = keywh.split('_')
                        w_r = int(w_r)
                        h_r = int(h_r)
                        w_r, h_r = resize_image(w_r, h_r, self.max_side[0],
                                                self.max_side[1])
                        h_r = max(
                            int(h_r // self.divided_factor[1] *
                                self.divided_factor[1]),
                            self.divided_factor[1])
                        w_r = max(
                            int(w_r // self.divided_factor[0] *
                                self.divided_factor[0]),
                            self.divided_factor[0])

                        key = str(w_r) + '_' + str(h_r)
                        if key in img_label_pair_list:
                            img_label_pair_list[key].extend(value_list)
                        else:
                            img_label_pair_list[key] = value_list

        self.need_reset = True
        self.img_label_pair_list = {}
        self.img_label_pair_list_small = {}
        for key in img_label_pair_list:
            json_data_list = img_label_pair_list[key]
            if self.mode == 'train':
                if len(json_data_list) < num_replicas * 2:
                    continue
                if len(json_data_list) <= 8 * num_replicas:
                    self.img_label_pair_list_small[key] = json_data_list
                # 补充至num_replicas的倍数
                fill_num = num_replicas - len(json_data_list) % num_replicas
                if fill_num < num_replicas:
                    for i in range(fill_num):
                        json_data_list.append(
                            json_data_list[i % len(json_data_list)])
                # 按照GPU数和rank划分数据
                json_data_list = json_data_list[rank::num_replicas]
                random.shuffle(json_data_list)
            self.img_label_pair_list[key] = json_data_list

        del img_label_pair_list
        self.ops = create_operators(dataset_config['transforms'],
                                    global_config)
        self.interpolation = T.InterpolationMode.BICUBIC
        transforms = []
        transforms.extend([
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])
        self.transforms = T.Compose(transforms)
        self.math_pattern = re.compile(r'\\\(|\\\[')
        self.rules = [
            # 超过4个连续 <|unk|> → 4 个空格
            (r'(?:<\|unk\|>){5,}', '    '),
            # 超过4个连续 \uffff → 4 个空格
            (r'(?:\uffff){5,}', '    '),
            #（可选）单个 <|unk|> → 空格
            (r'<\|unk\|>', ' '),
            #（可选）单个 \uffff → 空格
            (r'\uffff', ' '),
            (r'_{6,}', '______'),
            (r'\.{6,}', '......'),
        ]

    def __init_lmdb(self):
        """Initializes the LMDB environment."""
        if self.env is None:
            # Set max_readers high enough for potential multi-process data loading
            # map_size should be large enough to hold your entire dataset
            # self.env = lmdb.open(self.lmdb_path, readonly=True, create=False) # create=False 表示如果不存在则报错
            self.env = lmdb.open(
                self.lmdb_path,
                max_readers=32,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            self.txn = self.env.begin()

    def __init_line_lmdb(self, epoch_current):
        """Initializes the LMDB environment."""

        lmdb_paths = {
            'k12_': f'{self.root_path}/K-12/image_lmdb_qwen',
            'lsvt_': f'{self.root_path}/LSVT-2019/image_lmdb',
            'mtwi_': f'{self.root_path}/webdata_MTWI/image_lmdb',
            'tal_': f'{self.root_path}/TAL_OCR_HW/image_lmdb',
            'hwdb_': f'{self.root_path}/HWDB2Train/image_lmdb',
            'exam_': f'{self.root_path}/K-12_exam/image_lmdb_qwen',
            'page_hwpdf': f'{self.root_path}/hw_pdf/image_lmdb_qwen_tex_norm',
            'crop_hwpdf':
            f'{self.root_path}/hw_pdf/image_lmdb_qwen_crop_tex_norm',
            'xhs_hw_': f'{self.root_path}/hw_xhs/image_lmdb_qwen_tex_norm',  #
            'dfcf_pdf_':
            f'{self.root_path}/dfcf_pdf_dpi300/image_lmdb',  # fix huice
            'hauxuejiaoyu_': f'{self.root_path}/huaxue_jiaoyu/image_lmdb',
            'augtex_':
            f'{self.root_path}/latex_aug_new_circled/image_lmdb',  # rm only textcircled
            'nongminribao_pdf_':
            f'{self.root_path}/nongminribao_pdf_dpi300/image_lmdb',  #126963
            'hiertext_': f'{self.root_path}/hiertext_lmdb/image_lmdb',
        }

        if self.use_table:
            lmdb_paths[
                'hw_table_'] = f'{self.root_path}/hw_pdf_table/image_lmdb'
            lmdb_paths[
                'dfcf_table_'] = f'{self.root_path}/dfcf_table/image_lmdb_refine'
            lmdb_paths[
                'jiaoyu_table_'] = f'{self.root_path}/jiaoyu_table/image_lmdb'
            lmdb_paths[
                'pubtab1m_table_'] = f'{self.root_path}/pubtab1m_table/image_lmdb_refine'

        self.txns = {}
        lmdb_args = dict(max_readers=32,
                         readonly=True,
                         lock=False,
                         readahead=False,
                         meminit=False)

        for prefix, path in lmdb_paths.items():
            env = lmdb.open(path, **lmdb_args)
            self.txns[prefix] = env.begin()

    def crop_pdf_as_image(self, data_info, dpi=300, is_math=False):
        file_name = data_info['file_name']

        bbox_crop = data_info['bbox']
        if isinstance(bbox_crop[0], list):
            bbox = [
                bbox_crop[0][0], bbox_crop[0][1], bbox_crop[0][2],
                bbox_crop[0][3]
            ]
            for i in range(1, len(bbox_crop)):
                bbox[0] = min(bbox[0], bbox_crop[i][0])
                bbox[1] = min(bbox[1], bbox_crop[i][1])
                bbox[2] = max(bbox[2], bbox_crop[i][2])
                bbox[3] = max(bbox[3], bbox_crop[i][3])
        else:
            bbox = bbox_crop

        if self.env is not None:
            if '/home/ubuntu/bigdiskdata/' in file_name:  # for ch pdf
                doc = fitz.open(
                    file_name.replace('/home/ubuntu/bigdiskdata/',
                                      f'{self.root_path}/ch_pdf_label/') +
                    '.pdf')
            else:
                # Use LMDB to read the PDF file
                pdf_data = self.txn.get(file_name.encode('utf-8'))
                if pdf_data is None:
                    return None
                doc = fitz.open(stream=pdf_data, filetype='pdf')
        else:
            if '/home/ubuntu/bigdiskdata/' in file_name:  # for ch pdf
                doc = fitz.open(
                    file_name.replace('/home/ubuntu/bigdiskdata/',
                                      f'{self.root_path}/ch_pdf_label/') +
                    '.pdf')
            else:
                doc = fitz.open(self.root_path + file_name + '.pdf')
        # crop pdf with bbox
        page = doc[0]
        rect = fitz.Rect(*[x * 72. / dpi for x in bbox])
        if not is_math and random.random() < 0.2:
            text_dict = page.get_text('words', clip=rect)
            box_color = random_color()
            for x0, y0, x1, y1, text, _, _, _ in text_dict:
                if text.strip():  # 过滤空文本
                    text = re.sub(r'[,:;.]', '', text)
                    if text[1:-1].isdigit() and (
                            text[0] == '(' and text[-1] == ')'
                            or text[0] == '[' and text[-1] == ']'):
                        bbox_color = [x0 + 3, y0, x1 - 3, y1]
                        rect_bbox = fitz.Rect(*bbox_color)
                        page.draw_rect(rect_bbox, color=box_color, width=1.0)
                    elif random.random() < 0.01:
                        bbox_color = [x0, y0, x1, y1]
                        rect_bbox = fitz.Rect(*bbox_color)
                        page.draw_rect(rect_bbox, color=box_color, width=1.0)
        crop_img = page.get_pixmap(clip=rect, dpi=dpi)
        # 转换为img
        image = Image.frombytes('RGB', [crop_img.width, crop_img.height],
                                crop_img.samples)
        line_data = False
        if 'lines_bbox' in data_info:
            lines_bbox_ = data_info['lines_bbox']
            if data_info['class_name'] == 'region':
                lines_bbox = []
                for bbox_item in lines_bbox_:
                    lines_bbox.extend(bbox_item)
            else:
                lines_bbox = lines_bbox_
            if len(lines_bbox) > 3:
                line_data = True
            else:
                line_data = False
            if len(lines_bbox) > 0:
                np_img = np.array(image)
                zero_img = np.zeros_like(np_img) + 255
                for bbox_item in lines_bbox:
                    x1, y1, x2, y2 = bbox_item
                    x1 = x1 - bbox[0]
                    y1 = y1 - bbox[1]
                    x2 = x2 - bbox[0]
                    y2 = y2 - bbox[1]
                    zero_img[y1:y2, x1:x2] = np_img[y1:y2, x1:x2]
                image = Image.fromarray(zero_img)
        doc.close()
        return image, line_data

    def resize_norm_img(self,
                        data,
                        imgW,
                        imgH,
                        zoom_time,
                        padding=False,
                        line_data=False):
        img = data['image']

        zoom_time = float(zoom_time) / 10.

        if zoom_time <= 1.:
            zoom_time = 1.
        w, h = img.size
        if self.use_zoom:
            if imgW / self.divided_factor[
                    0] >= 5 or imgH / self.divided_factor[1] >= 5:
                zoom_time = min(imgW / self.divided_factor[0] - 2,
                                imgH / self.divided_factor[1] - 2, zoom_time)
            else:
                zoom_time = min(imgW / self.divided_factor[0] - 1,
                                imgH / self.divided_factor[1] - 1, zoom_time)
            if imgW >= self.max_side[0] or imgH >= self.max_side[1]:
                zoom_time = max(zoom_time, 1.0)
            else:
                zoom_time = max(zoom_time, 2.0)
            # if not line_data else max(zoom_time, 1.0)
            imgW_r = imgW / float(zoom_time)
            imgH_r = imgH / float(zoom_time)
            imgW_r = max(
                int(imgW_r // self.divided_factor[0] * self.divided_factor[0]),
                64)
            imgH_r = max(
                int(imgH_r // self.divided_factor[1] * self.divided_factor[1]),
                64)
            resized_image = F.resize(img, (imgH_r, imgW_r),
                                     interpolation=self.interpolation)
        else:
            resized_image = F.resize(img, (imgH, imgW),
                                     interpolation=self.interpolation)

        img = self.transforms(resized_image)
        valid_ratio = min(1.0, float(w / imgW))
        data['image'] = img
        data['valid_ratio'] = valid_ratio
        return data

    def remove_space_before_sn(self, text, rep_str):
        # 匹配 “汉字 + 空格 + <|sn|>” 这种模式
        # \u4e00-\u9fff 是中文字符的 Unicode 范围
        return re.sub(r'([\u4e00-\u9fff])\s*<\|sn\|>', r'\1' + rep_str, text)

    def clean_label(self, text):
        for rule in self.rules:
            text = re.sub(rule[0], rule[1], text)
        text = fix_diacritics_regex(text)
        return text

    def __getitem__(self, properties):

        if len(properties) != 5:
            img_id, w_r, h_r, zoom_time = properties
        else:
            img_id, w_r, h_r, zoom_time, resume_batch = properties
            if resume_batch > 0:
                return np.zeros((1, 3), dtype=np.float32)
        key = str(w_r) + '_' + str(h_r)
        if img_id > len(self.img_label_pair_list[key]) - 1:
            data_info = self.img_label_pair_list_small[key][img_id]
        else:
            data_info = self.img_label_pair_list[key][img_id]
        label = data_info['label']
        if isinstance(label, list):
            label = '\n\n'.join(label)
        if self.add_return:
            rep_str = '<|sn|>'
        else:
            rep_str = ''
        label = label.replace('<<<hyphen>>>', '')
        label = label.replace('<<<change_line_token_wrap>>>', rep_str)
        label = label.replace('<<<change_line_token_split>>>', rep_str)
        label = label.replace('<<<null>>>', '')
        label = self.remove_space_before_sn(label, rep_str)
        label = self.clean_label(label)
        if not self.add_return:
            label = label.replace('<|sn|>', '')
        try:
            file_name = data_info['file_name']
            img_data = None
            if self.test_data:
                image, line_data = self.crop_pdf_as_image(
                    data_info,
                    dpi=300,
                    is_math=self.math_pattern.search(label))
            else:
                for prefix, txn in self.txns.items():
                    if file_name.startswith(prefix):
                        img_data = txn.get(file_name.encode('utf-8'))
                        break
                if img_data is not None or 'bbox' not in data_info:
                    image = Image.open(io.BytesIO(img_data)).convert('RGB')
                    line_data = True
                else:
                    image, line_data = self.crop_pdf_as_image(
                        data_info,
                        dpi=300,
                        is_math=self.math_pattern.search(label))
            if image is None or label is None:
                if len(self.img_label_pair_list[key]) <= 8:
                    rnd_properties = [
                        random.randint(
                            0,
                            len(self.img_label_pair_list_small[key]) - 1), w_r,
                        h_r, zoom_time
                    ]
                else:
                    rnd_properties = [
                        random.randint(0,
                                       len(self.img_label_pair_list[key]) - 1),
                        w_r, h_r, zoom_time
                    ]
                return self.__getitem__(rnd_properties)
            if self.use_table and (file_name.startswith('table_')
                                   or file_name.startswith('dfcf_table_')):
                line_data = False
            data = {'image': image, 'label': label, 'arxiv': not line_data}
            data = transform(data, self.ops[:-1])
            if data is None:
                if len(self.img_label_pair_list[key]) <= 8:
                    rnd_properties = [
                        random.randint(
                            0,
                            len(self.img_label_pair_list_small[key]) - 1), w_r,
                        h_r, zoom_time
                    ]
                else:
                    rnd_properties = [
                        random.randint(0,
                                       len(self.img_label_pair_list[key]) - 1),
                        w_r, h_r, zoom_time
                    ]
                return self.__getitem__(rnd_properties)

            if self.use_aug:
                w, h = image.size
                if w > self.max_side[0] or h > self.max_side[1]:
                    w, h = resize_image(w, h, self.max_side[0],
                                        self.max_side[1])
                    h = max(
                        int(h // self.divided_factor[1] *
                            self.divided_factor[1]), self.divided_factor[1])
                    w = max(
                        int(w // self.divided_factor[0] *
                            self.divided_factor[0]), self.divided_factor[0])
                    data['image'] = data['image'].resize((w, h))

                if any(
                        file_name.startswith(prefix) for prefix in [
                            'lsvt_', 'mtwi_', 'tal_', 'page_hwpdf',
                            'crop_hwpdf', 'xhs_hw_', 'hiertext_', 'hw_table_'
                        ]):
                    data['arxiv'] = False
                else:
                    data['arxiv'] = True
                data = self.docaug(data)

            data = self.resize_norm_img(data,
                                        w_r,
                                        h_r,
                                        zoom_time,
                                        line_data=line_data)
            outs = transform(data, self.ops[-1:])
        except:
            self.logger.error(
                'When parsing line {}, error happened with msg: {}'.format(
                    data_info['file_name'], traceback.format_exc()))
            outs = None
        if outs is None:
            # during evaluation, we should fix the idx to get same results for many times of evaluation.
            if len(self.img_label_pair_list[key]) <= 8:
                rnd_properties = [
                    random.randint(
                        0,
                        len(self.img_label_pair_list_small[key]) - 1), w_r,
                    h_r, zoom_time
                ]
            else:
                rnd_properties = [
                    random.randint(0,
                                   len(self.img_label_pair_list[key]) - 1),
                    w_r, h_r, zoom_time
                ]
            return self.__getitem__(rnd_properties)
        return outs

    def __len__(self):
        len_all = 0
        for key in self.img_label_pair_list:
            len_all += len(self.img_label_pair_list[key])
        return len_all
