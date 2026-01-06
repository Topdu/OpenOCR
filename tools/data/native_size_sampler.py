import numpy as np
import torch
from torch.utils.data import Sampler


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


class NaSizeSampler(Sampler):

    def __init__(
            self,
            data_source,
            max_side=[64 * 15, 64 * 22],  # w,h
            min_bs=1,
            max_bs=1024,
            resume_iter=0,
            scale_ratio=2,
            seed=None):
        """
            multi scale samper
            Args:
                data_source(dataset)
                scales(list): several scales for image resolution
                first_bs(int): batch size for the first scale in scales
                divided_factor(list[w, h]): ImageNet models down-sample images by a factor, ensure that width and height dimensions are multiples are multiple of devided_factor.
                is_training(boolean): mode
        """
        self.data_source = data_source

        self.seed = data_source.seed

        self.img_label_pair_list = data_source.img_label_pair_list
        self.shuffle = data_source.do_shuffle
        self.is_training = data_source.mode == 'train'

        max_side = data_source.max_side
        batch_list = []
        sorted_keys = sorted(
            self.img_label_pair_list.keys(),
            key=lambda k: int(k.split('_')[0]) * int(k.split('_')[1]))
        for key in sorted_keys:
            w_r, h_r = key.split('_')
            w_r = int(w_r)
            h_r = int(h_r)

            current_bs = int(((max_side[0] * max_side[1]) // (w_r * h_r)) *
                             min_bs * scale_ratio)
            current_bs = min(current_bs, max_bs,
                             len(self.img_label_pair_list[key]))
            bacth_num = len(self.img_label_pair_list[key]) // current_bs
            current_img_indices_all = np.arange(len(
                self.img_label_pair_list[key]),
                                                dtype=np.int64)

            drop = len(self.img_label_pair_list[key]) - current_bs * bacth_num
            if self.is_training and drop > 0:
                drop_full_num = current_bs - drop
                drop_full = np.random.choice(current_img_indices_all,
                                             drop_full_num,
                                             replace=True)
                current_img_indices = np.append(current_img_indices_all,
                                                drop_full)
            else:
                current_img_indices = current_img_indices_all[:bacth_num *
                                                              current_bs]
            current_batch_list = current_img_indices.reshape(-1, current_bs, 1)
            w_r_batch = np.full_like(current_batch_list, w_r)
            h_r_batch = np.full_like(current_batch_list, h_r)
            random_zoom_time = np.random.randint(
                -5, 50, [current_batch_list.shape[0], 1, 1])
            random_zoom_time = np.tile(random_zoom_time,
                                       (1, current_batch_list.shape[1], 1))
            current_batch_list = np.concatenate(
                [current_batch_list, w_r_batch, h_r_batch, random_zoom_time],
                axis=-1)
            batch_list.extend(current_batch_list.tolist())

            if not self.is_training and drop > 0:
                current_img_indices = current_img_indices_all[bacth_num *
                                                              current_bs:]
                current_batch_list = current_img_indices.reshape(-1, drop, 1)
                w_r_batch = np.full_like(current_batch_list, w_r)
                h_r_batch = np.full_like(current_batch_list, h_r)
                random_zoom_time = np.random.randint(
                    -5, 50, [current_batch_list.shape[0], 1, 1])
                random_zoom_time = np.tile(random_zoom_time,
                                           (1, current_batch_list.shape[1], 1))
                current_batch_list = np.concatenate([
                    current_batch_list, w_r_batch, h_r_batch, random_zoom_time
                ],
                                                    axis=-1)
                batch_list.extend(current_batch_list.tolist())

        self.fix_cobatch = 4
        self.batch_list = batch_list  # [[[img_id, w_r, h_r, zoom_time], ...], ...]
        self.length = len(self.batch_list)
        self.batchs_id_sort = [i for i in range(self.length)]
        self.batchs_in_one_epoch_id = self.batchs_id_sort.copy()
        self.is_shuffled = False
        self.resume_iter = resume_iter
        if self.shuffle or self.is_training:
            g = torch.Generator()
            g.manual_seed(self.seed)  # 让所有进程的种子相同
            random_indices = torch.randperm(len(self.batchs_in_one_epoch_id),
                                            generator=g).tolist()
            self.batchs_in_one_epoch_id = [
                self.batchs_in_one_epoch_id[i] for i in random_indices
            ]
            if self.resume_iter > 0:
                # resume iter
                for iter_ in range(len(self.batch_list)):
                    if iter_ <= self.resume_iter:
                        batch_list_current = self.batch_list[
                            self.batchs_in_one_epoch_id[iter_]]
                        batch_list_current_resume = []
                        for batch in batch_list_current:
                            batch.append(1)
                            batch_list_current_resume.append(batch)
                        self.batch_list[self.batchs_in_one_epoch_id[
                            iter_]] = batch_list_current_resume
                    else:
                        batch_list_current = self.batch_list[
                            self.batchs_in_one_epoch_id[iter_]]
                        batch_list_current_resume = []
                        for batch in batch_list_current:
                            batch.append(0)
                            batch_list_current_resume.append(batch)
                        self.batch_list[self.batchs_in_one_epoch_id[
                            iter_]] = batch_list_current_resume
                self.resume_iter = 0

    def __iter__(self):
        for batch_tuple_id in self.batchs_in_one_epoch_id:
            yield self.batch_list[batch_tuple_id]

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.length
