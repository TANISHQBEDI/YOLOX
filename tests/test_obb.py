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
    NUM_OBB_LABELS,
    angle_loss,
    canonicalize_rbox,
    decode_angle,
    encode_angle,
    ensure_theta_column,
    hflip_obb_label,
    kfiou_loss,
    kfiou_matrix,
    oriented_xyxy_theta_to_aabb,
    rbox_to_aabb_xyxy,
    rotate_obb_label,
    wrap_angle,
    yolov8_poly_to_rbox,
)
from yolox.utils.boxes import bboxes_iou, postprocess
from yolox.utils.checkpoint import torch_load


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


class TestCanonicalizeAndKFIoU(unittest.TestCase):
    def test_long_edge_is_width(self):
        cx, cy, w, h, theta = canonicalize_rbox(0.0, 0.0, 10.0, 40.0, 0.0)
        self.assertGreaterEqual(w, h)
        self.assertAlmostEqual(w, 40.0)
        self.assertAlmostEqual(h, 10.0)

    def test_identical_boxes_center_loss_is_zero(self):
        box = torch.tensor([[50.0, 50.0, 40.0, 20.0, 0.3]])
        loss = kfiou_loss(box, box)[0]
        # Volume KFIoU of identical Gaussians is 1/3, so 1-KFIoU=2/3; xy term is 0.
        self.assertAlmostEqual(float(loss), 2.0 / 3.0, places=3)

    def test_angle_mismatch_lowers_kfiou(self):
        a = torch.tensor([[50.0, 50.0, 40.0, 12.0, 0.0]])
        b = torch.tensor([[50.0, 50.0, 40.0, 12.0, math.pi / 2]])
        same = float(kfiou_matrix(a, a)[0, 0])
        rotated = float(kfiou_matrix(a, b)[0, 0])
        self.assertGreater(same, rotated)

    def test_far_centers_cost_more(self):
        a = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]])
        b = torch.tensor([[80.0, 80.0, 10.0, 10.0, 0.0]])
        self.assertGreater(float(kfiou_loss(a, b)[0]), float(kfiou_loss(a, a)[0]))

    def test_kfiou_finite_for_large_fp16_boxes(self):
        a = torch.tensor([[200.0, 200.0, 800.0, 80.0, 0.3]], dtype=torch.float16)
        b = torch.tensor([[210.0, 190.0, 790.0, 90.0, 0.2]], dtype=torch.float16)
        loss = kfiou_loss(a, b)
        self.assertTrue(torch.isfinite(loss).all())


class TestWrapAndPad(unittest.TestCase):
    def test_wrap_angle_range(self):
        for raw in (-math.pi, -2.0, 0.0, 2.0, math.pi, 3.0):
            w = wrap_angle(raw)
            self.assertGreater(w, -math.pi / 2.0)
            self.assertLessEqual(w, math.pi / 2.0)

    def test_ensure_theta_column_pads_aabb(self):
        labels = np.array([[1, 2, 3, 4, 0]], dtype=np.float32)
        out = ensure_theta_column(labels)
        self.assertEqual(out.shape, (1, NUM_OBB_LABELS))
        self.assertEqual(out[0, 5], 0.0)

    def test_ensure_theta_column_empty(self):
        out = ensure_theta_column(np.zeros((0, 5), dtype=np.float32))
        self.assertEqual(out.shape[1], NUM_OBB_LABELS)

    def test_angle_loss_zero_when_encoded_gt(self):
        theta = torch.tensor([0.4, -0.2])
        s, c = encode_angle(theta)
        loss = angle_loss(s, c, theta)
        self.assertTrue(torch.allclose(loss, torch.zeros_like(loss), atol=1e-6))


class TestRboxHull(unittest.TestCase):
    def test_rbox_hull_matches_xyxy_helper(self):
        rbox = torch.tensor([[50.0, 40.0, 30.0, 10.0, 0.6]])
        hull = rbox_to_aabb_xyxy(rbox)[0]
        x1 = rbox[0, 0] - rbox[0, 2] * 0.5
        y1 = rbox[0, 1] - rbox[0, 3] * 0.5
        x2 = rbox[0, 0] + rbox[0, 2] * 0.5
        y2 = rbox[0, 1] + rbox[0, 3] * 0.5
        xyxy = torch.tensor([[x1, y1, x2, y2]])
        hull2 = oriented_xyxy_theta_to_aabb(xyxy, rbox[:, 4])[0]
        self.assertTrue(torch.allclose(hull, hull2, atol=1e-5))

    def test_kfiou_matrix_shape_and_empty(self):
        a = torch.tensor([[1.0, 2.0, 8.0, 4.0, 0.1], [3.0, 4.0, 6.0, 5.0, -0.2]])
        b = torch.tensor([[1.0, 2.0, 8.0, 4.0, 0.1]])
        mat = kfiou_matrix(a, b)
        self.assertEqual(tuple(mat.shape), (2, 1))
        empty = kfiou_matrix(a.new_zeros((0, 5)), b)
        self.assertEqual(tuple(empty.shape), (0, 1))
        empty_loss = kfiou_loss(a.new_zeros((0, 5)), a.new_zeros((0, 5)))
        self.assertEqual(tuple(empty_loss.shape), (0,))

    def test_canonicalize_tensor_batch(self):
        cx = torch.tensor([0.0, 1.0])
        cy = torch.tensor([0.0, 1.0])
        w = torch.tensor([10.0, 8.0])
        h = torch.tensor([40.0, 3.0])
        th = torch.tensor([0.0, 0.1])
        _, _, w2, h2, _ = canonicalize_rbox(cx, cy, w, h, th)
        self.assertTrue(torch.all(w2 >= h2))

    def test_poly_rotated_square_has_finite_theta(self):
        s = math.sqrt(2.0) * 10
        poly = [50, 50 - s, 50 + s, 50, 50, 50 + s, 50 - s, 50]
        cx, cy, w, h, theta, aabb = yolov8_poly_to_rbox(poly, 100, 100)
        self.assertTrue(np.isfinite([cx, cy, w, h, theta]).all())
        self.assertGreaterEqual(w, h)
        self.assertGreater(aabb[2] - aabb[0], 0)

    def test_hull_iou_identical_is_one(self):
        rbox = torch.tensor([[40.0, 40.0, 30.0, 10.0, 0.7]])
        hull = rbox_to_aabb_xyxy(rbox)
        iou = bboxes_iou(hull, hull, xyxy=True)
        self.assertAlmostEqual(float(iou[0, 0]), 1.0, places=5)

    def test_matching_hull_not_raw_xywh(self):
        # A 45° thin box's hull is much larger than treating (w,h) as AABB.
        rbox = torch.tensor([[50.0, 50.0, 40.0, 4.0, math.pi / 4]])
        hull = rbox_to_aabb_xyxy(rbox)[0]
        naive_w = float(hull[2] - hull[0])
        naive_h = float(hull[3] - hull[1])
        self.assertGreater(naive_w, 40.0 * 0.7)
        self.assertGreater(naive_h, 4.0 * 2.0)


class TestPostprocessOBB(unittest.TestCase):
    def _pred(self, cx, cy, w, h, theta, obj=0.95, cls=0.95):
        s, c = encode_angle(torch.tensor(theta))
        return [cx, cy, w, h, obj, cls, float(s), float(c)]

    def test_duplicate_rotated_boxes_nms(self):
        row = self._pred(32.0, 32.0, 20.0, 8.0, 0.4)
        pred = torch.tensor([[row, row]], dtype=torch.float32)
        out = postprocess(pred, num_classes=1, conf_thre=0.25, nms_thre=0.3, class_agnostic=True)
        self.assertIsNotNone(out[0])
        self.assertEqual(out[0].shape[0], 1)
        self.assertGreater(out[0].shape[1], 7)

    def test_empty_after_conf(self):
        row = self._pred(32.0, 32.0, 20.0, 8.0, 0.0, obj=0.01, cls=0.01)
        pred = torch.tensor([[row]], dtype=torch.float32)
        out = postprocess(pred, num_classes=1, conf_thre=0.5, nms_thre=0.45)
        self.assertIsNone(out[0])

    def test_decoded_theta_roundtrip(self):
        theta = 0.35
        pred = torch.tensor([[self._pred(10.0, 12.0, 16.0, 6.0, theta)]], dtype=torch.float32)
        out = postprocess(pred, num_classes=1, conf_thre=0.1, nms_thre=0.9)
        self.assertIsNotNone(out[0])
        self.assertEqual(out[0].shape[0], 1)
        self.assertAlmostEqual(float(out[0][0, 7]), theta, places=4)

    def test_separated_boxes_survive_nms(self):
        a = self._pred(16.0, 16.0, 10.0, 6.0, 0.2)
        b = self._pred(80.0, 80.0, 10.0, 6.0, -0.4)
        pred = torch.tensor([[a, b]], dtype=torch.float32)
        out = postprocess(pred, num_classes=1, conf_thre=0.25, nms_thre=0.3)
        self.assertEqual(out[0].shape[0], 2)


class TestTorchLoad(unittest.TestCase):
    def test_torch_load_roundtrip(self, tmp_path=None):
        import tempfile

        payload = {"model": {"a": torch.ones(2)}, "start_epoch": 3}
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name
        torch.save(payload, path)
        loaded = torch_load(path, map_location="cpu")
        self.assertEqual(loaded["start_epoch"], 3)
        self.assertTrue(torch.equal(loaded["model"]["a"], torch.ones(2)))
        os.remove(path)


class TestHeadKFIoUBackward(unittest.TestCase):
    def test_iou_loss_finite_with_obb_labels(self):
        from yolox.exp import get_exp

        exp = get_exp(exp_name="yolox-s")
        model = exp.get_model()
        model.train()
        imgs = torch.randn(1, 3, 64, 64)
        labels = torch.zeros(1, 4, 6)
        labels[0, 0] = torch.tensor([0.0, 32.0, 32.0, 20.0, 10.0, 0.5])
        out = model(imgs, labels)
        for key in ("iou_loss", "angle_loss", "conf_loss", "cls_loss", "total_loss"):
            self.assertIn(key, out)
            self.assertTrue(torch.isfinite(out[key]), msg=key)
        out["total_loss"].backward()
        grad_ok = any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for n, p in model.named_parameters()
            if "reg_preds" in n or "angle_preds" in n
        )
        self.assertTrue(grad_ok)


class TestLossWeights(unittest.TestCase):
    def test_exp_copies_weights_onto_head(self):
        from yolox.exp import get_exp

        exp = get_exp(exp_name="yolox-s")
        exp.reg_weight = 1.0
        exp.angle_weight = 5.0
        model = exp.get_model()
        self.assertEqual(model.head.reg_weight, 1.0)
        self.assertEqual(model.head.angle_weight, 5.0)

    def test_pallets_exp_keeps_detection_defaults(self):
        from yolox.exp import get_exp

        exp = get_exp("exps/example/custom/yolox_s_pallets.py", None)
        self.assertEqual(exp.reg_weight, 5.0)
        self.assertEqual(exp.angle_weight, 0.5)
        model = exp.get_model()
        self.assertEqual(model.head.reg_weight, 5.0)
        self.assertEqual(model.head.angle_weight, 0.5)


class TestAffineOBB(unittest.TestCase):
    def test_90_deg_keeps_long_edge_as_width(self):
        import cv2
        from yolox.data.data_augment import apply_affine_to_bboxes

        M = cv2.getRotationMatrix2D((100.0, 100.0), 90.0, 1.0)
        targets = np.array([[80.0, 90.0, 120.0, 110.0, 0.0, 0.0]], dtype=np.float32)
        out = apply_affine_to_bboxes(targets, (200, 200), M, scale=1.0, angle_deg=90)
        self.assertEqual(len(out), 1)
        w = float(out[0, 2] - out[0, 0])
        h = float(out[0, 3] - out[0, 1])
        self.assertGreaterEqual(w, h)
        self.assertAlmostEqual(w, 40.0, delta=1.0)
        self.assertAlmostEqual(h, 20.0, delta=1.0)
        self.assertLess(abs(abs(math.degrees(float(out[0, 5]))) - 90.0), 3.0)

    def test_shear_refits_angle_not_just_center(self):
        import cv2
        from yolox.data.data_augment import apply_affine_to_bboxes

        R = cv2.getRotationMatrix2D(angle=0.0, center=(0.0, 0.0), scale=1.0)
        M = np.ones((2, 3), dtype=np.float64)
        shear_x = math.tan(20.0 * math.pi / 180.0)
        M[0] = R[0]
        M[1] = R[1] + shear_x * R[0]
        M[0, 2] = 0.0
        M[1, 2] = 0.0
        targets = np.array([[80.0, 90.0, 120.0, 110.0, 0.0, 0.0]], dtype=np.float32)
        out = apply_affine_to_bboxes(targets, (200, 200), M)
        self.assertEqual(len(out), 1)
        w = float(out[0, 2] - out[0, 0])
        h = float(out[0, 3] - out[0, 1])
        self.assertGreaterEqual(w, h)
        self.assertTrue(np.isfinite(out).all())
        # Old path added angle_deg=0 and left theta=0; shear must rotate the box.
        self.assertGreater(abs(float(out[0, 5])), 0.05)

    def test_numpy_canonicalize_batch(self):
        cx = np.array([0.0, 1.0], dtype=np.float32)
        cy = np.array([0.0, 1.0], dtype=np.float32)
        w = np.array([10.0, 8.0], dtype=np.float32)
        h = np.array([40.0, 3.0], dtype=np.float32)
        th = np.array([0.0, 0.1], dtype=np.float32)
        _, _, w2, h2, _ = canonicalize_rbox(cx, cy, w, h, th)
        self.assertTrue(np.all(w2 >= h2))


class TestRotatedEval(unittest.TestCase):
    def test_identical_rbox_iou_is_one(self):
        from yolox.evaluators.rotated_eval import box_iou_rotated

        box = np.array([[50.0, 40.0, 30.0, 10.0, 0.4]])
        iou = box_iou_rotated(box, box)
        self.assertAlmostEqual(float(iou[0, 0]), 1.0, places=3)

    def test_perfect_match_ap50_and_recall(self):
        from yolox.evaluators.rotated_eval import eval_rotated_ap

        gt = {0: np.array([[50.0, 50.0, 40.0, 20.0, 0.3, 1.0]])}
        preds = [{
            "image_id": 0,
            "category_id": 1,
            "rbox": [50.0, 50.0, 40.0, 20.0, 0.3],
            "score": 0.9,
        }]
        ap, rec = eval_rotated_ap(gt, preds, iou_thr=0.5)
        self.assertAlmostEqual(ap, 1.0, places=3)
        self.assertAlmostEqual(rec, 1.0, places=3)

    def test_missing_pred_recall_is_zero(self):
        from yolox.evaluators.rotated_eval import eval_rotated_ap

        gt = {0: np.array([[50.0, 50.0, 40.0, 20.0, 0.0, 1.0]])}
        ap, rec = eval_rotated_ap(gt, [], iou_thr=0.5)
        self.assertEqual(ap, 0.0)
        self.assertEqual(rec, 0.0)

    def test_angle_error_hurts_rotated_iou_more_than_aabb(self):
        from yolox.evaluators.rotated_eval import box_iou_rotated
        from yolox.utils.boxes import bboxes_iou

        gt = torch.tensor([[50.0, 50.0, 40.0, 8.0, math.radians(30.0)]])
        pred = torch.tensor([[50.0, 50.0, 40.0, 8.0, 0.0]])
        aabb_iou = float(
            bboxes_iou(rbox_to_aabb_xyxy(gt), rbox_to_aabb_xyxy(pred), xyxy=True)[0, 0]
        )
        r_iou = float(box_iou_rotated(gt.numpy(), pred.numpy())[0, 0])
        self.assertGreater(aabb_iou, r_iou)


class TestRotatedAssign(unittest.TestCase):
    def test_rbox_iou_identity_is_one(self):
        from yolox.utils.obb import rbox_iou

        box = torch.tensor([[50.0, 40.0, 30.0, 10.0, 0.4]])
        iou = rbox_iou(box, box)
        self.assertAlmostEqual(float(iou[0, 0]), 1.0, places=3)

    def test_matching_iou_penalizes_wrong_angle(self):
        from yolox.utils.obb import rbox_iou

        gt = torch.tensor([[50.0, 50.0, 40.0, 8.0, math.radians(30.0)]])
        pred_ok = torch.tensor([[50.0, 50.0, 40.0, 8.0, math.radians(30.0)]])
        pred_flat = torch.tensor([[50.0, 50.0, 40.0, 8.0, 0.0]])
        hull_flat = float(
            bboxes_iou(rbox_to_aabb_xyxy(gt), rbox_to_aabb_xyxy(pred_flat), xyxy=True)[0, 0]
        )
        rot_ok = float(rbox_iou(gt, pred_ok)[0, 0])
        rot_flat = float(rbox_iou(gt, pred_flat)[0, 0])
        self.assertGreater(rot_ok, 0.9)
        self.assertGreater(hull_flat, rot_flat)
        self.assertGreater(rot_ok, rot_flat)


if __name__ == "__main__":
    unittest.main()
