#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Rotated-IoU AP / recall for OBB detections.

AABB COCO AP stays the detection KPI (finding the pallet). This metric
grades the actual oriented box: a level prediction on a tilted GT is a
miss at IoU 0.5 even if the hull overlap looks fine.

IoU uses torchvision.ops.box_iou_rotated when available (same (cx,cy,w,h,
degrees) convention as nms_rotated). OpenCV rotatedRectangleIntersection
is the fallback so eval still runs without that op.
"""

import math
from collections import defaultdict

import numpy as np

try:
    import torch
    import torchvision

    _HAS_TV_ROTATED_IOU = hasattr(torchvision.ops, "box_iou_rotated")
except Exception:
    torch = None
    torchvision = None
    _HAS_TV_ROTATED_IOU = False

import cv2


def _cv2_rbox_iou(a, b):
    """Single-pair IoU. a, b are (cx, cy, w, h, angle_deg)."""
    r1 = ((float(a[0]), float(a[1])), (float(a[2]), float(a[3])), float(a[4]))
    r2 = ((float(b[0]), float(b[1])), (float(b[2]), float(b[3])), float(b[4]))
    ret, pts = cv2.rotatedRectangleIntersection(r1, r2)
    if ret == 0 or pts is None:
        return 0.0
    inter = cv2.contourArea(pts)
    union = float(a[2]) * float(a[3]) + float(b[2]) * float(b[3]) - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def box_iou_rotated(boxes_a, boxes_b):
    """
    Pairwise rotated IoU.

    boxes_*: (N, 5) / (M, 5) as cx, cy, w, h, theta_radians.
    Returns (N, M) float64.
    """
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 5)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 5)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)

    if _HAS_TV_ROTATED_IOU:
        try:
            ta = torch.as_tensor(a, dtype=torch.float32)
            tb = torch.as_tensor(b, dtype=torch.float32)
            deg_a = torch.stack(
                (ta[:, 0], ta[:, 1], ta[:, 2], ta[:, 3], ta[:, 4] * 180.0 / math.pi), dim=1
            )
            deg_b = torch.stack(
                (tb[:, 0], tb[:, 1], tb[:, 2], tb[:, 3], tb[:, 4] * 180.0 / math.pi), dim=1
            )
            return torchvision.ops.box_iou_rotated(deg_a, deg_b).cpu().numpy().astype(np.float64)
        except Exception:
            pass

    ious = np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    a_deg = np.concatenate((a[:, :4], a[:, 4:5] * 180.0 / math.pi), axis=1)
    b_deg = np.concatenate((b[:, :4], b[:, 4:5] * 180.0 / math.pi), axis=1)
    for i, ra in enumerate(a_deg):
        for j, rb in enumerate(b_deg):
            ious[i, j] = _cv2_rbox_iou(ra, rb)
    return ious


def _coco_ap_and_recall(tp, fp, n_gt):
    """101-point interpolated AP (COCO) and recall at the end of the ranked list."""
    if n_gt <= 0:
        return 0.0, 0.0
    if tp.size == 0:
        return 0.0, 0.0
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    recalls = tp_c / float(n_gt)
    precisions = tp_c / np.maximum(tp_c + fp_c, 1e-16)
    rec_thrs = np.linspace(0.0, 1.0, 101)
    sampled = []
    for t in rec_thrs:
        above = precisions[recalls >= t]
        sampled.append(float(above.max()) if above.size else 0.0)
    return float(np.mean(sampled)), float(recalls[-1])


def eval_rotated_ap(gt_by_image, pred_list, iou_thr=0.5, max_dets=100):
    """
    Mean (over classes with GT) COCO-style AP at a single rotated-IoU threshold.

    gt_by_image: image_id -> (N, 6) [cx, cy, w, h, theta, category_id]
    pred_list: dicts with image_id, category_id, rbox [cx,cy,w,h,theta], score
    """
    cat_ids = set()
    for arr in gt_by_image.values():
        if arr is None or len(arr) == 0:
            continue
        cat_ids.update(int(c) for c in np.asarray(arr)[:, 5])
    for p in pred_list:
        cat_ids.add(int(p["category_id"]))
    if not cat_ids:
        return 0.0, 0.0

    aps, recalls = [], []
    for cat in sorted(cat_ids):
        n_gt = 0
        gt_map = {}
        for img_id, arr in gt_by_image.items():
            arr = np.asarray(arr).reshape(-1, 6) if arr is not None and len(arr) else np.zeros((0, 6))
            mask = arr[:, 5] == cat if len(arr) else np.zeros((0,), dtype=bool)
            boxes = arr[mask][:, :5] if mask.size else np.zeros((0, 5))
            gt_map[img_id] = {
                "boxes": boxes,
                "matched": np.zeros((boxes.shape[0],), dtype=bool),
            }
            n_gt += boxes.shape[0]
        if n_gt == 0:
            continue

        class_preds = [p for p in pred_list if int(p["category_id"]) == cat]
        class_preds.sort(key=lambda p: p["score"], reverse=True)
        per_image = defaultdict(int)
        ranked = []
        for p in class_preds:
            img_id = int(p["image_id"])
            per_image[img_id] += 1
            if per_image[img_id] <= max_dets:
                ranked.append(p)

        tp = np.zeros(len(ranked), dtype=np.float64)
        fp = np.zeros(len(ranked), dtype=np.float64)
        for i, p in enumerate(ranked):
            rec = gt_map.get(int(p["image_id"]))
            if rec is None or rec["boxes"].shape[0] == 0:
                fp[i] = 1.0
                continue
            ious = box_iou_rotated(np.asarray(p["rbox"], dtype=np.float64), rec["boxes"])[0]
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr and not rec["matched"][j]:
                tp[i] = 1.0
                rec["matched"][j] = True
            else:
                fp[i] = 1.0
        ap, rec = _coco_ap_and_recall(tp, fp, n_gt)
        aps.append(ap)
        recalls.append(rec)

    if not aps:
        return 0.0, 0.0
    return float(np.mean(aps)), float(np.mean(recalls))


def summarize_rotated_metrics(gt_by_image, pred_list):
    ap50, rec50 = eval_rotated_ap(gt_by_image, pred_list, iou_thr=0.5)
    ap75, _ = eval_rotated_ap(gt_by_image, pred_list, iou_thr=0.75)
    info = (
        "\nRotated IoU (oriented box, not AABB hull):\n"
        " Average Precision  (AP) @[ IoU=0.50      | area=all | maxDets=100 ] = {:.3f}\n"
        " Average Precision  (AP) @[ IoU=0.75      | area=all | maxDets=100 ] = {:.3f}\n"
        " Recall @ IoU=0.50 (after conf / NMS) = {:.3f}\n"
    ).format(ap50, ap75, rec50)
    return {"ap50": ap50, "ap75": ap75, "recall50": rec50, "info": info}


def gt_rboxes_from_dataset(dataset):
    """image_id -> (N,6) original-pixel rboxes, or None if the dataset is AABB-only."""
    ds = dataset
    while hasattr(ds, "_dataset"):
        ds = ds._dataset
    if not hasattr(ds, "gt_rboxes"):
        return None
    boxes = ds.gt_rboxes
    ids = getattr(ds, "ids", range(len(boxes)))
    return {int(img_id): np.asarray(arr) for img_id, arr in zip(ids, boxes)}
