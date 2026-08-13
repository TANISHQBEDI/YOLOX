#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os

from yolox.data.datasets.yolov8_obb import default_pallets_dir
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # YOLOv8 OBB pallets dataset lives one directory above the YOLOX repo
        self.data_dir = default_pallets_dir()
        self.train_split = "train"
        self.val_split = "valid"
        self.num_classes = 1

        # Roboflow export is 416x416
        self.input_size = (416, 416)
        self.test_size = (416, 416)
        self.random_size = (10, 16)

        self.max_epoch = 100
        self.data_num_workers = 4
        self.eval_interval = 5

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        from yolox.data import TrainTransform, YoloV8OBBDataset

        return YoloV8OBBDataset(
            data_dir=self.data_dir,
            split=self.train_split,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import ValTransform, YoloV8OBBDataset

        legacy = kwargs.get("legacy", False)
        return YoloV8OBBDataset(
            data_dir=self.data_dir,
            split=self.val_split,
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )
