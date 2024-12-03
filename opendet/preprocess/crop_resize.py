import cv2


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
