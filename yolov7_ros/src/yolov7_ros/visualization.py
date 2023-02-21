import sys
import random
import pathlib

import numpy as np

# add yolov7 submodule to path
FILE_ABS_DIR = pathlib.Path(__file__).absolute().parent
YOLOV7_ROOT = (FILE_ABS_DIR / 'yolov7').as_posix()
if YOLOV7_ROOT not in sys.path:
    sys.path.append(YOLOV7_ROOT)

from utils.plots import plot_one_box

class Visualizer:
    def __init__(self, detector, line_thickness=1):
        self.model = detector.model
        self.__names = self.model.names
        self.__colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.__names]
        self.__line_thickness = line_thickness

    def draw_2d_bboxes(self, img, dets):
        for det in dets:
            *xyxy, conf, cls = det
            label = f'{self.__names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, img, label=label, color=self.__colors[int(cls)],
                        line_thickness=self.__line_thickness)
