#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Unit tests for OBB angle encode/decode, aug, and YOLOX head wiring."""

import math
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yolox.utils.obb import (
    decode_angle,
    encode_angle,
    hflip_obb_label,
    oriented_xyxy_theta_to_aabb,
    rotate_obb_label,
    yolov8_poly_to_rbox,
)


class TestOBBAngle(unittest.TestCase):
    def test_encode_decode_roundtrip(self):
        for theta_deg in [-89, -45, 0, 1, 45, 89]:
            theta = torch.tensor(math.radians(theta_deg))
            s, c = encode_angle(theta)
            recovered = decode_angle(s, c)
            self.assertLess(
                abs(math.degrees(recovered.item()) - theta_deg),
                1e-3,
                f"roundtrip failed for {theta_deg} degrees, "
                f"got {math.degrees(recovered.item())}",
            )

        theta = torch.tensor(math.radians(90))
        s, c = encode_angle(theta)
        recovered_deg = math.degrees(decode_angle(s, c).item())
        self.assertTrue(
            abs(recovered_deg - 90) < 1e-3 or abs(recovered_deg + 90) < 1e-3,
            f"90-degree boundary case failed, got {recovered_deg}",
        )

    def test_90_degree_rotation_swaps_orientation(self):
        cx, cy, w, h, theta = 100, 100, 50, 20, 0.0
        img_w, img_h = 200, 200
        _, _, _, _, new_theta = rotate_obb_label(
            cx, cy, w, h, theta, angle_deg=90, img_w=img_w, img_h=img_h
        )
        self.assertLess(abs(abs(math.degrees(new_theta)) - 90), 1e-3)

    def test_hflip_negates_angle(self):
        cx, cy, w, h, theta = 50, 100, 40, 15, math.radians(30)
        img_w = 200
        new_cx, _, _, _, new_theta = hflip_obb_label(cx, cy, w, h, theta, img_w)
        self.assertEqual(new_cx, img_w - cx)
        self.assertLess(abs(math.degrees(new_theta) - (-30)), 1e-3)


class TestYOLOXAngleHead(unittest.TestCase):
    def test_head_predicts_sin_cos_and_trains(self):
        from yolox.exp import get_exp

        exp = get_exp(exp_name="yolox-s")
        model = exp.get_model()
        model.train()

        imgs = torch.randn(2, 3, 64, 64)
        labels = torch.zeros(2, 8, 6)
        labels[0, 0] = torch.tensor([0.0, 32.0, 32.0, 16.0, 10.0, 0.4])
        labels[1, 0] = torch.tensor([1.0, 20.0, 40.0, 12.0, 8.0, -0.3])

        outputs = model(imgs, labels)
        self.assertIn("angle_loss", outputs)
        self.assertIn("total_loss", outputs)
        self.assertTrue(torch.isfinite(outputs["angle_loss"]))
        outputs["total_loss"].backward()

        # angle branch exists and received gradients
        angle_grad = False
        for name, param in model.named_parameters():
            if "angle_preds" in name and param.grad is not None:
                angle_grad = True
                break
        self.assertTrue(angle_grad, "angle_preds did not receive gradients")

    def test_inference_output_has_angle_channels(self):
        from yolox.exp import get_exp

        exp = get_exp(exp_name="yolox-s")
        model = exp.get_model()
        model.eval()
        imgs = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            outputs = model(imgs)
        # cx, cy, w, h, obj, cls * num_classes, sin2theta, cos2theta
        self.assertEqual(outputs.shape[-1], 7 + exp.num_classes)


class TestYoloV8OBBDataset(unittest.TestCase):
    def test_poly_to_rbox_axis_aligned(self):
        poly = [30, 40, 70, 40, 70, 60, 30, 60]  # 40x20 axis-aligned
        cx, cy, w, h, theta, aabb = yolov8_poly_to_rbox(poly, 100, 100)
        self.assertAlmostEqual(cx, 50, places=3)
        self.assertAlmostEqual(cy, 50, places=3)
        self.assertAlmostEqual(max(w, h), 40, places=3)
        self.assertAlmostEqual(min(w, h), 20, places=3)
        self.assertLess(abs(math.degrees(theta)) % 90, 1e-2)

    def test_load_pallets_valid_sample(self):
        from yolox.data import TrainTransform, YoloV8OBBDataset
        from yolox.data.datasets.yolov8_obb import default_pallets_dir

        data_dir = default_pallets_dir()
        if not os.path.isdir(os.path.join(data_dir, "valid", "images")):
            self.skipTest(f"pallets dataset not found at {data_dir}")

        dataset = YoloV8OBBDataset(
            data_dir=data_dir,
            split="valid",
            img_size=(416, 416),
            preproc=TrainTransform(max_labels=50, flip_prob=0.0, hsv_prob=0.0),
        )
        self.assertGreater(len(dataset), 0)
        self.assertEqual(dataset._classes, ("Pallet-Detection",))
        img, labels, img_info, img_id = dataset[0]
        self.assertEqual(img.shape[0], 3)
        self.assertEqual(labels.shape[1], 6)
        n = int((labels.sum(axis=1) > 0).sum())
        self.assertGreaterEqual(n, 0)
        if n > 0:
            self.assertTrue(np.isfinite(labels[:n, 5]).all())


class TestAABBfromOBB(unittest.TestCase):
    def test_zero_angle_is_identity(self):
        xyxy = torch.tensor([[10.0, 20.0, 30.0, 50.0]])
        theta = torch.tensor([0.0])
        aabb = oriented_xyxy_theta_to_aabb(xyxy, theta)
        self.assertTrue(torch.allclose(aabb, xyxy, atol=1e-5))

    def test_90_deg_swaps_extents(self):
        # oriented 20x40 box at (50, 50)
        xyxy = torch.tensor([[40.0, 30.0, 60.0, 70.0]])
        theta = torch.tensor([math.pi / 2])
        aabb = oriented_xyxy_theta_to_aabb(xyxy, theta)[0]
        # hull should be 40x20
        self.assertAlmostEqual(float(aabb[2] - aabb[0]), 40.0, places=4)
        self.assertAlmostEqual(float(aabb[3] - aabb[1]), 20.0, places=4)


if __name__ == "__main__":
    unittest.main()
