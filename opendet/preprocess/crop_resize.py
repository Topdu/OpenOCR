import cv2
import numpy as np


def padding_image(img, size=(640, 640)):
    """
    Padding an image using OpenCV:
    - If the image is smaller than the target size, pad it to 640x640.
    - If the image is larger than the target size, split it into multiple 640x640 images and record positions.

    :param image_path: Path to the input image.
    :param output_dir: Directory to save the output images.
    :param size: The target size for padding or splitting (default 640x640).
    :return: List of tuples containing the coordinates of the top-left corner of each cropped 640x640 image.
    """

    img_height, img_width = img.shape[:2]
    target_width, target_height = size

    # If image is smaller than target size, pad the image to 640x640

    # Calculate padding amounts (top, bottom, left, right)
    pad_top = 0
    pad_bottom = target_height - img_height
    pad_left = 0
    pad_right = target_width - img_width

    # Pad the image (white padding, border type: constant)
    padded_img = cv2.copyMakeBorder(img,
                                    pad_top,
                                    pad_bottom,
                                    pad_left,
                                    pad_right,
                                    cv2.BORDER_CONSTANT,
                                    value=[0, 0, 0])

    # Return the padded area positions (top-left and bottom-right coordinates of the original image)
    return padded_img


def is_poly_outside_rect(poly, x, y, w, h):
    poly = np.array(poly)
    if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
        return True
    if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
        return True
    return False


def split_regions(axis):
    regions = []
    min_axis = 0
    for i in range(1, axis.shape[0]):
        if axis[i] != axis[i - 1] + 1:
            region = axis[min_axis:i]
            min_axis = i
            regions.append(region)
    return regions


def random_select(axis, max_size):
    xx = np.random.choice(axis, size=2)
    xmin = np.min(xx)
    xmax = np.max(xx)
    xmin = np.clip(xmin, 0, max_size - 1)
    xmax = np.clip(xmax, 0, max_size - 1)
    return xmin, xmax


def region_wise_random_select(regions, max_size):
    selected_index = list(np.random.choice(len(regions), 2))
    selected_values = []
    for index in selected_index:
        axis = regions[index]
        xx = int(np.random.choice(axis, size=1))
        selected_values.append(xx)
    xmin = min(selected_values)
    xmax = max(selected_values)
    return xmin, xmax


def crop_area(im, text_polys, min_crop_side_ratio, max_tries):
    h, w, _ = im.shape
    h_array = np.zeros(h, dtype=np.int32)
    w_array = np.zeros(w, dtype=np.int32)
    for points in text_polys:
        points = np.round(points, decimals=0).astype(np.int32)
        minx = np.min(points[:, 0])
        maxx = np.max(points[:, 0])
        w_array[minx:maxx] = 1
        miny = np.min(points[:, 1])
        maxy = np.max(points[:, 1])
        h_array[miny:maxy] = 1
    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]

    if len(h_axis) == 0 or len(w_axis) == 0:
        return 0, 0, w, h

    h_regions = split_regions(h_axis)
    w_regions = split_regions(w_axis)

    for i in range(max_tries):
        if len(w_regions) > 1:
            xmin, xmax = region_wise_random_select(w_regions, w)
        else:
            xmin, xmax = random_select(w_axis, w)
        if len(h_regions) > 1:
            ymin, ymax = region_wise_random_select(h_regions, h)
        else:
            ymin, ymax = random_select(h_axis, h)

        if (xmax - xmin < min_crop_side_ratio * w or ymax - ymin < min_crop_side_ratio * h):
            # area too small
            continue
        num_poly_in_rect = 0
        for poly in text_polys:
            if not is_poly_outside_rect(poly, xmin, ymin, xmax - xmin,
                                        ymax - ymin):
                num_poly_in_rect += 1
                break

        if num_poly_in_rect > 0:
            return xmin, ymin, xmax - xmin, ymax - ymin

    return 0, 0, w, h


class EastRandomCropData(object):

    def __init__(
        self,
        size=(640, 640),
        max_tries=10,
        min_crop_side_ratio=0.1,
        keep_ratio=True,
        **kwargs,
    ):
        self.size = size
        self.max_tries = max_tries
        self.min_crop_side_ratio = min_crop_side_ratio
        self.keep_ratio = keep_ratio

    def __call__(self, data):
        img = data['image']
        text_polys = data['polys']
        ignore_tags = data['ignore_tags']
        texts = data['texts']
        all_care_polys = [
            text_polys[i] for i, tag in enumerate(ignore_tags) if not tag
        ]
        # 计算crop区域
        crop_x, crop_y, crop_w, crop_h = crop_area(img, all_care_polys,
                                                   self.min_crop_side_ratio,
                                                   self.max_tries)
        # crop 图片 保持比例填充
        scale_w = self.size[0] / crop_w
        scale_h = self.size[1] / crop_h
        scale = min(scale_w, scale_h)
        h = int(crop_h * scale)
        w = int(crop_w * scale)
        if self.keep_ratio:
            padimg = np.zeros((self.size[1], self.size[0], img.shape[2]),
                              img.dtype)
            padimg[:h, :w] = cv2.resize(
                img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))
            img = padimg
        else:
            img = cv2.resize(
                img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w],
                tuple(self.size),
            )
        # crop 文本框
        text_polys_crop = []
        ignore_tags_crop = []
        texts_crop = []
        for poly, text, tag in zip(text_polys, texts, ignore_tags):
            poly = ((poly - (crop_x, crop_y)) * scale).tolist()
            if not is_poly_outside_rect(poly, 0, 0, w, h):
                text_polys_crop.append(poly)
                ignore_tags_crop.append(tag)
                texts_crop.append(text)
        data['image'] = img
        data['polys'] = np.array(text_polys_crop)
        data['ignore_tags'] = ignore_tags_crop
        data['texts'] = texts_crop
        return data


class CropResize(object):

    def __init__(self, size=(640, 640), interpolation=cv2.INTER_LINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, data):
        """
        Resize an image using OpenCV:
        - If the image is smaller than the target size, pad it to 640x640.
        - If the image is larger than the target size, split it into multiple 640x640 images and record positions.

        :param image_path: Path to the input image.
        :param output_dir: Directory to save the output images.
        :param size: The target size for padding or splitting (default 640x640).
        :return: List of tuples containing the coordinates of the top-left corner of each cropped 640x640 image.
        """
        img = data['image']
        img_height, img_width = img.shape[:2]
        target_width, target_height = self.size

        # If image is smaller than target size, pad the image to 640x640
        if img_width <= target_width and img_height <= target_height:
            # Calculate padding amounts (top, bottom, left, right)
            if img_width == target_width and img_height == target_height:
                return [img], [[0, 0, img_width, img_height]]
            padded_img = padding_image(img, self.size)

            # Return the padded area positions (top-left and bottom-right coordinates of the original image)
            return [padded_img], [[0, 0, img_width, img_height]]

        if img_width < target_width:
            img = cv2.copyMakeBorder(img,
                                     0,
                                     0,
                                     0,
                                     target_width - img_width,
                                     cv2.BORDER_CONSTANT,
                                     value=[0, 0, 0])

        if img_height < target_height:
            img = cv2.copyMakeBorder(img,
                                     0,
                                     target_height - img_height,
                                     0,
                                     0,
                                     cv2.BORDER_CONSTANT,
                                     value=[0, 0, 0])
            # raise ValueError("Image dimensions must be greater than or equal to target size")

        img_height, img_width = img.shape[:2]
        # If image is larger than or equal to target size, crop it into 640x640 tiles
        crop_positions = []
        count = 0
        cropped_img_list = []
        for top in range(0, img_height - target_height // 2,
                         target_height // 2):
            for left in range(0, img_width - target_height // 2,
                              target_width // 2):
                # Calculate the bottom and right boundaries for the crop
                right = min(left + target_width, img_width)
                bottom = min(top + target_height, img_height)
                if right > img_width:
                    right = img_width
                    left = max(0, right - target_width)
                if bottom > img_height:
                    bottom = img_height
                    top = max(0, bottom - target_height)
                # Crop the image
                cropped_img = img[top:bottom, left:right]
                if bottom - top < target_height or right - left < target_width:
                    cropped_img = padding_image(cropped_img, self.size)

                count += 1
                cropped_img_list.append(cropped_img)

                # Record the position of the cropped image
                crop_positions.append([left, top, right, bottom])

        # print(f"Images cropped and saved at {output_dir}.")

        return cropped_img_list, crop_positions
