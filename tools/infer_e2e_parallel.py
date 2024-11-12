from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
import queue
import os
import sys
import time

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import numpy as np
import cv2
import json
from PIL import Image
from tools.utils.utility import get_image_file_list, check_and_read
from tools.infer_rec import OpenRecognizer
from tools.infer_det import OpenDetector
from tools.engine import Config
from tools.infer.utility import get_rotate_crop_image, get_minarea_rect_crop, draw_ocr_box_txt


class OpenOCRWithSingleDetector:

    def __init__(self,
                 cfg_det,
                 cfg_rec,
                 drop_score=0.5,
                 det_box_type='quad',
                 max_rec_threads=4):
        self.text_detector = OpenDetector(cfg_det)
        self.text_recognizer = OpenRecognizer(cfg_rec)
        self.det_box_type = det_box_type
        self.drop_score = drop_score
        self.queue = queue.Queue(
        )  # Queue to hold detected boxes for recognition
        self.results = {}
        self.lock = threading.Lock()  # Lock for thread-safe access to results
        self.max_rec_threads = max_rec_threads
        self.stop_signal = threading.Event()  # Signal to stop threads

    def start_recognition_threads(self):
        """Start recognition threads."""
        self.rec_threads = []
        for _ in range(self.max_rec_threads):
            t = threading.Thread(target=self.recognize_text)
            t.start()
            self.rec_threads.append(t)

    def detect_text(self, image_list):
        """Single-threaded text detection for all images."""
        for image_id, (img_numpy, ori_img) in enumerate(image_list):
            dt_boxes = self.text_detector(img_numpy=img_numpy)[0]['boxes']
            if dt_boxes is None:
                self.results[image_id] = []  # If no boxes, set empty results
                continue

            dt_boxes = sorted_boxes(dt_boxes)
            for box in dt_boxes:
                tmp_box = np.array(box).astype(np.float32)
                img_crop = (get_rotate_crop_image(ori_img, tmp_box)
                            if self.det_box_type == 'quad' else
                            get_minarea_rect_crop(ori_img, tmp_box))
                self.queue.put(
                    (image_id, box, img_crop)
                )  # Put image ID, detected box, and cropped image in queue

        # Signal that no more items will be added to the queue
        self.stop_signal.set()

    def recognize_text(self):
        """Recognize text in each cropped image."""
        while not self.stop_signal.is_set() or not self.queue.empty():
            try:
                image_id, box, img_crop = self.queue.get(timeout=5)
                rec_result = self.text_recognizer(img_numpy_list=[img_crop])[0]
                text, score = rec_result['text'], rec_result['score']
                if score >= self.drop_score:
                    with self.lock:
                        # Ensure results dictionary has a list for each image ID
                        if image_id not in self.results:
                            self.results[image_id] = []
                        self.results[image_id].append({
                            'transcription':
                            text,
                            'points':
                            np.array(box).tolist(),
                            'score':
                            score
                        })
                self.queue.task_done()
            except queue.Empty:
                continue

    def process_images(self, image_list):
        """Process a list of images."""
        # Initialize results dictionary
        self.results = {i: [] for i in range(len(image_list))}

        # Start recognition threads
        self.start_recognition_threads()

        # Start detection in the main thread
        self.detect_text(image_list)

        # Wait for recognition threads to finish
        for t in self.rec_threads:
            t.join()

        return self.results


def sorted_boxes(dt_boxes):
    """Sort text boxes top-to-bottom, left-to-right."""
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    return sorted_boxes


def main(cfg_det, cfg_rec):
    image_file_list = get_image_file_list('./testA/')
    drop_score = 0.5
    text_sys = OpenOCRWithSingleDetector(cfg_det,
                                         cfg_rec,
                                         drop_score=drop_score)
    is_visualize = False
    font_path = '/path/doc/fonts/simfang.ttf'
    draw_img_save_dir = './testA_repvitdet_svtrv2_rec_parallel/'
    os.makedirs(draw_img_save_dir, exist_ok=True)
    save_results = []

    # Prepare images
    images = []
    t_start = time.time()
    for image_file in image_file_list:
        img, flag_gif, flag_pdf = check_and_read(image_file)
        if not flag_gif and not flag_pdf:
            img = cv2.imread(image_file)
        if img is not None:
            images.append((img, img.copy()))

    results = text_sys.process_images(images)
    print(f'time cost: {time.time() - t_start}')
    # Save results and visualize
    for image_id, res in results.items():
        image_file = image_file_list[image_id]
        save_pred = f'{os.path.basename(image_file)}\t{json.dumps(res, ensure_ascii=False)}\n'
        # print(save_pred)
        save_results.append(save_pred)

        if is_visualize:
            dt_boxes = [result['points'] for result in res]
            rec_res = [result['transcription'] for result in res]
            rec_score = [result['score'] for result in res]
            image = Image.fromarray(
                cv2.cvtColor(images[image_id][0], cv2.COLOR_BGR2RGB))
            draw_img = draw_ocr_box_txt(image,
                                        dt_boxes,
                                        rec_res,
                                        rec_score,
                                        drop_score=drop_score,
                                        font_path=font_path)

            save_file = os.path.join(draw_img_save_dir,
                                     os.path.basename(image_file))
            cv2.imwrite(save_file, draw_img[:, :, ::-1])

    with open(os.path.join(draw_img_save_dir, 'system_results.txt'),
              'w',
              encoding='utf-8') as f:
        f.writelines(save_results)


if __name__ == '__main__':
    cfg_det = Config('./configs/det/dbnet/repvit_db.yml')
    cfg_rec = Config('./configs/rec/svtrv2/svtrv2_ch.yml')
    main(cfg_det.cfg, cfg_rec.cfg)
