#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""YOLOv8 OBB dataset (Roboflow-style: split/images + split/labels)."""

import copy
import math
import os
from pathlib import Path

import cv2
import numpy as np

from yolox.utils.obb import NUM_OBB_LABELS, wrap_angle, yolov8_poly_to_rbox

from .datasets_wrapper import CacheDataset, cache_read_img

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def default_pallets_dir():
    """../pallets relative to the YOLOX repo root."""
    import yolox

    yolox_root = os.path.dirname(os.path.dirname(yolox.__file__))
    return os.path.abspath(os.path.join(yolox_root, "..", "pallets"))


def _load_class_names(data_dir):
    yaml_path = os.path.join(data_dir, "data.yaml")
    names = {}
    if not os.path.isfile(yaml_path):
        return ("Pallet-Detection",)
    in_names = False
    with open(yaml_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("names:"):
                rest = line[len("names:"):].strip()
                in_names = True
                if rest.startswith("[") and rest.endswith("]"):
                    inner = rest[1:-1]
                    return tuple(x.strip().strip("'\"") for x in inner.split(",") if x.strip())
                continue
            if in_names:
                if ":" not in line:
                    break
                key, val = line.split(":", 1)
                if not key.strip().isdigit():
                    break
                names[int(key.strip())] = val.strip().strip("'\"")
    if names:
        return tuple(names[i] for i in sorted(names))
    return ("Pallet-Detection",)


def _image_hw(path):
    try:
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
        return h, w
    except Exception:
        img = cv2.imread(path)
        assert img is not None, f"file named {path} not found"
        return img.shape[0], img.shape[1]


def _parse_yolov8_obb_line(parts, img_w, img_h):
    """
    Parse one label line.

    Returns (obb_xyxy_theta, aabb_xywh_cls) or None.
    obb: [x1, y1, x2, y2, cls, theta] using oriented w/h as unrotated extents.
    aabb: [x, y, w, h, cls] envelope of the four corners, original pixels.
    """
    if len(parts) < 6:
        return None
    cls = int(float(parts[0]))
    if len(parts) >= 9:
        cx, cy, bw, bh, theta, aabb = yolov8_poly_to_rbox(parts[1:9], img_w, img_h)
    elif len(parts) >= 6:
        cx = float(parts[1]) * img_w
        cy = float(parts[2]) * img_h
        bw = float(parts[3]) * img_w
        bh = float(parts[4]) * img_h
        theta = float(parts[5])
        if abs(theta) > math.pi:
            theta = math.radians(theta)
        theta = wrap_angle(theta)
        aabb = np.array(
            [cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5],
            dtype=np.float32,
        )
    else:
        return None
    if bw <= 1 or bh <= 1:
        return None
    obb = [
        cx - bw * 0.5,
        cy - bh * 0.5,
        cx + bw * 0.5,
        cy + bh * 0.5,
        cls,
        theta,
    ]
    aabb_xywh = [
        float(aabb[0]),
        float(aabb[1]),
        float(aabb[2] - aabb[0]),
        float(aabb[3] - aabb[1]),
        cls,
    ]
    return obb, aabb_xywh


class YoloV8OBBDataset(CacheDataset):
    """
    Load a YOLOv8 oriented-box dataset.

    Expected layout (Roboflow / Ultralytics)::

        data_dir/
          data.yaml
          train/images/*.jpg
          train/labels/*.txt
          valid/images/*.jpg
          valid/labels/*.txt

    Each label line is ``class x1 y1 x2 y2 x3 y3 x4 y4`` with normalized corners.
    """

    def __init__(
        self,
        data_dir=None,
        split="train",
        img_size=(416, 416),
        preproc=None,
        cache=False,
        cache_type="ram",
    ):
        if data_dir is None:
            data_dir = default_pallets_dir()
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.preproc = preproc
        self._classes = _load_class_names(data_dir)
        self.class_ids = list(range(1, len(self._classes) + 1))
        self.cats = [
            {"id": cid, "name": name}
            for cid, name in zip(self.class_ids, self._classes)
        ]

        img_dir = os.path.join(data_dir, split, "images")
        assert os.path.isdir(img_dir), f"image directory not found: {img_dir}"

        self.img_files = sorted(
            f for f in os.listdir(img_dir)
            if Path(f).suffix.lower() in IMAGE_EXTS
        )
        self.ids = list(range(len(self.img_files)))
        self.num_imgs = len(self.img_files)
        assert self.num_imgs > 0, f"no images found in {img_dir}"

        self.rel_paths = [os.path.join(split, "images", f) for f in self.img_files]
        self.annotations = [self.load_anno_from_ids(i) for i in self.ids]
        self.coco = self._build_coco()

        super().__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=data_dir,
            cache_dir_name=f"cache_{split}",
            path_filename=self.rel_paths,
            cache=cache,
            cache_type=cache_type,
        )

    def __len__(self):
        return self.num_imgs

    def load_anno_from_ids(self, index):
        file_name = self.img_files[index]
        img_path = os.path.join(self.data_dir, self.split, "images", file_name)
        height, width = _image_hw(img_path)

        label_path = os.path.join(
            self.data_dir, self.split, "labels", Path(file_name).stem + ".txt"
        )
        objs = []
        aabb_xywh = []
        if os.path.isfile(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    parsed = _parse_yolov8_obb_line(parts, width, height)
                    if parsed is None:
                        continue
                    obb, aabb = parsed
                    objs.append(obb)
                    aabb_xywh.append(aabb)

        num_objs = len(objs)
        res = np.zeros((num_objs, NUM_OBB_LABELS), dtype=np.float32)
        gt_rboxes = np.zeros((num_objs, 6), dtype=np.float32)
        if num_objs:
            res[:, :] = np.asarray(objs, dtype=np.float32)
            gt_rboxes[:, 0] = (res[:, 0] + res[:, 2]) * 0.5
            gt_rboxes[:, 1] = (res[:, 1] + res[:, 3]) * 0.5
            gt_rboxes[:, 2] = res[:, 2] - res[:, 0]
            gt_rboxes[:, 3] = res[:, 3] - res[:, 1]
            gt_rboxes[:, 4] = res[:, 5]
            gt_rboxes[:, 5] = [
                self.class_ids[int(c)] for c in res[:, 4]
            ]

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))
        return (res, img_info, resized_info, file_name, aabb_xywh, gt_rboxes)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_image(self, index):
        img_file = os.path.join(self.data_dir, self.rel_paths[index])
        img = cv2.imread(img_file)
        assert img is not None, f"file named {img_file} not found"
        return img

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    @cache_read_img(use_cache=True)
    def read_img(self, index):
        return self.load_resized_img(index)

    def pull_item(self, index):
        label, origin_image_size, _, _, _, _ = self.annotations[index]
        img = self.read_img(index)
        return img, copy.deepcopy(label), origin_image_size, np.array([self.ids[index]])

    @property
    def gt_rboxes(self):
        """Original-pixel rboxes [cx, cy, w, h, theta, category_id] per image."""
        return [ann[5] for ann in self.annotations]

    @CacheDataset.mosaic_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)
        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id

    def _build_coco(self):
        from pycocotools.coco import COCO

        images = []
        annotations = []
        categories = [
            {"id": cid, "name": name, "supercategory": "object"}
            for cid, name in zip(self.class_ids, self._classes)
        ]
        ann_id = 1
        for img_id, rec in enumerate(self.annotations):
            _, img_info, _, file_name, aabb_xywh = rec[:5]
            height, width = img_info
            images.append({
                "id": img_id,
                "file_name": file_name,
                "height": int(height),
                "width": int(width),
            })
            for box in aabb_xywh:
                x, y, w, h, cls = box
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": self.class_ids[int(cls)],
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "area": float(max(w, 0.0) * max(h, 0.0)),
                    "iscrowd": 0,
                    "segmentation": [],
                })
                ann_id += 1

        coco = COCO()
        coco.dataset = {
            "info": {"description": "YOLOv8 OBB (AABB eval)"},
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        coco.createIndex()
        return coco
