#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Oriented bounding box helpers for YOLOX.

Angle representation: predict (sin(2*theta), cos(2*theta)) instead of raw theta
to avoid the 0/180-degree wraparound discontinuity. Doubling the angle also
handles 180-degree box symmetry.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

import cv2

__all__ = [
    "NUM_OBB_LABELS",
    "wrap_angle",
    "encode_angle",
    "decode_angle",
    "angle_loss",
    "ensure_theta_column",
    "rotate_obb_label",
    "hflip_obb_label",
    "yolov8_poly_to_rbox",
    "oriented_xyxy_theta_to_aabb",
    "canonicalize_rbox",
    "kfiou_loss",
    "kfiou_matrix",
    "rbox_to_aabb_xyxy",
]

# Dataset labels: [x1, y1, x2, y2, cls, theta]
# Training targets after TrainTransform: [cls, cx, cy, w, h, theta]
NUM_OBB_LABELS = 6


def wrap_angle(theta):
    """Wrap theta in radians into (-pi/2, pi/2]."""
    if isinstance(theta, np.ndarray):
        return np.mod(theta + np.pi / 2.0, np.pi) - np.pi / 2.0
    if torch.is_tensor(theta):
        return torch.remainder(theta + math.pi / 2.0, math.pi) - math.pi / 2.0
    return (theta + math.pi / 2.0) % math.pi - math.pi / 2.0


def encode_angle(theta):
    """Ground-truth theta (radians) -> (sin(2*theta), cos(2*theta)) target."""
    two_theta = 2.0 * theta
    return torch.sin(two_theta), torch.cos(two_theta)


def decode_angle(sin2theta, cos2theta):
    """
    Recover theta in radians, range (-pi/2, pi/2], from predicted
    sin(2*theta) / cos(2*theta).
    """
    norm = torch.sqrt(sin2theta ** 2 + cos2theta ** 2).clamp(min=1e-6)
    sin2theta_n = sin2theta / norm
    cos2theta_n = cos2theta / norm
    two_theta = torch.atan2(sin2theta_n, cos2theta_n)  # (-pi, pi]
    return two_theta / 2.0  # (-pi/2, pi/2]


def angle_loss(pred_sin2t, pred_cos2t, target_theta):
    """
    Smooth-L1 on the (sin2theta, cos2theta) pair.

    pred_sin2t, pred_cos2t: (N,) raw predictions for matched positive anchors
    target_theta: (N,) ground truth angle in radians for the same anchors

    Returns per-sample loss of shape (N,) so callers can sum / num_fg.
    """
    target_sin2t, target_cos2t = encode_angle(target_theta)
    loss_sin = F.smooth_l1_loss(pred_sin2t, target_sin2t, reduction="none")
    loss_cos = F.smooth_l1_loss(pred_cos2t, target_cos2t, reduction="none")
    return loss_sin + loss_cos


def ensure_theta_column(labels):
    """Pad [x1, y1, x2, y2, cls] labels with theta=0 if needed."""
    if labels is None or len(labels) == 0:
        return np.zeros((0, NUM_OBB_LABELS), dtype=np.float32)
    labels = np.asarray(labels)
    if labels.ndim != 2:
        return labels
    if labels.shape[1] >= NUM_OBB_LABELS:
        return labels
    pad = np.zeros((labels.shape[0], NUM_OBB_LABELS - labels.shape[1]), dtype=labels.dtype)
    return np.hstack((labels, pad))


def rotate_obb_label(cx, cy, w, h, theta, angle_deg, img_w, img_h):
    """
    Rotate a single OBB label by angle_deg (counter-clockwise, degrees) around
    the image center, matching an image rotation of the same angle.

    Returns: new_cx, new_cy, w, h, new_theta
    """
    angle_rad = math.radians(angle_deg)
    icx, icy = img_w / 2.0, img_h / 2.0

    dx, dy = cx - icx, cy - icy
    new_dx = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
    new_dy = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
    new_cx, new_cy = icx + new_dx, icy + new_dy

    new_theta = wrap_angle(theta + angle_rad)
    return canonicalize_rbox(new_cx, new_cy, w, h, new_theta)


def hflip_obb_label(cx, cy, w, h, theta, img_w):
    """Horizontal flip: mirror center x, negate theta."""
    new_cx = img_w - cx
    new_theta = wrap_angle(-theta)
    return canonicalize_rbox(new_cx, cy, w, h, new_theta)


def yolov8_poly_to_rbox(poly, img_w, img_h):
    """
    Convert a YOLOv8 OBB 4-corner polygon to (cx, cy, w, h, theta) plus AABB.

    poly: 8 values [x1, y1, x2, y2, x3, y3, x4, y4], normalized (0-1) or pixels.
    img_w, img_h: original image size in pixels.

    Returns:
        cx, cy, w, h, theta: oriented box in pixels, theta in radians (-pi/2, pi/2]
        aabb: [x1, y1, x2, y2] axis-aligned envelope in pixels
    """
    pts = np.asarray(poly, dtype=np.float64).reshape(4, 2)
    if pts.max() <= 1.01:
        pts = pts.copy()
        pts[:, 0] *= img_w
        pts[:, 1] *= img_h

    aabb = np.array(
        [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()],
        dtype=np.float32,
    )
    (cx, cy), (bw, bh), angle_deg = cv2.minAreaRect(pts.astype(np.float32))
    theta = wrap_angle(math.radians(float(angle_deg)))
    cx, cy, bw, bh, theta = canonicalize_rbox(float(cx), float(cy), float(bw), float(bh), float(theta))
    return cx, cy, bw, bh, theta, aabb


def oriented_xyxy_theta_to_aabb(xyxy, theta):
    """
    Axis-aligned hull of an oriented box.

    xyxy is (N, 4) built from oriented (w, h) as if unrotated: the visual /
    training extents. theta is (N,) radians. Used for NMS and COCO eval so
    those steps match the GT AABB envelope.
    """
    x1, y1, x2, y2 = xyxy.unbind(-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    hw = (x2 - x1) * 0.5
    hh = (y2 - y1) * 0.5
    c = torch.cos(theta).abs()
    s = torch.sin(theta).abs()
    aabb_hw = c * hw + s * hh
    aabb_hh = s * hw + c * hh
    return torch.stack(
        (cx - aabb_hw, cy - aabb_hh, cx + aabb_hw, cy + aabb_hh), dim=-1
    )


def canonicalize_rbox(cx, cy, w, h, theta):
    """Force long-edge width (w >= h) and wrap theta to (-pi/2, pi/2]."""
    if torch.is_tensor(cx):
        swap = h > w
        w_new = torch.where(swap, h, w)
        h_new = torch.where(swap, w, h)
        theta = wrap_angle(torch.where(swap, theta + math.pi / 2.0, theta))
        return cx, cy, w_new, h_new, theta
    if h > w:
        w, h = h, w
        theta = theta + math.pi / 2.0
    return cx, cy, w, h, wrap_angle(theta)


def _det2(mat):
    return mat[..., 0, 0] * mat[..., 1, 1] - mat[..., 0, 1] * mat[..., 1, 0]


def _inv2(mat, eps=1e-6):
    a, b = mat[..., 0, 0], mat[..., 0, 1]
    c, d = mat[..., 1, 0], mat[..., 1, 1]
    det = (a * d - b * c).clamp(min=eps)
    row0 = torch.stack((d / det, -b / det), dim=-1)
    row1 = torch.stack((-c / det, a / det), dim=-1)
    return torch.stack((row0, row1), dim=-2)


def _xywhr_to_sigma(xywhr):
    """(N, 5) cx,cy,w,h,theta -> center (N,2), covariance (N,2,2)."""
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-3)
    r = xywhr[..., 4]
    cos_r, sin_r = torch.cos(r), torch.sin(r)
    r00 = (cos_r * wh[..., 0] * 0.5) ** 2 + (sin_r * wh[..., 1] * 0.5) ** 2
    r11 = (sin_r * wh[..., 0] * 0.5) ** 2 + (cos_r * wh[..., 1] * 0.5) ** 2
    r01 = cos_r * sin_r * ((wh[..., 0] * 0.5) ** 2 - (wh[..., 1] * 0.5) ** 2)
    row0 = torch.stack((r00, r01), dim=-1)
    row1 = torch.stack((r01, r11), dim=-1)
    sigma = torch.stack((row0, row1), dim=-2)
    return xy, sigma


def kfiou_matrix(boxes_a, boxes_b, eps=1e-6):
    """
    Pairwise KFIoU in [0, 1]. boxes_* are (N,5) / (M,5) as cx,cy,w,h,theta.
    """
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return boxes_a.new_zeros((boxes_a.shape[0], boxes_b.shape[0]))
    _, sig_a = _xywhr_to_sigma(boxes_a)
    _, sig_b = _xywhr_to_sigma(boxes_b)
    sig_a = sig_a[:, None]
    sig_b = sig_b[None]
    vb_a = (4.0 * _det2(sig_a).clamp(min=0.0).sqrt()).clamp(min=eps)
    vb_b = (4.0 * _det2(sig_b).clamp(min=0.0).sqrt()).clamp(min=eps)
    sig_sum = sig_a + sig_b
    kalman = torch.matmul(sig_a, _inv2(sig_sum))
    sig_kf = sig_a - torch.matmul(kalman, sig_a)
    vb = 4.0 * _det2(sig_kf).clamp(min=0.0).sqrt()
    vb = torch.nan_to_num(vb, nan=0.0)
    return vb / (vb_a + vb_b - vb + eps)


def kfiou_loss(pred, target, beta=1.0 / 9.0, eps=1e-6):
    """
    Aligned KFIoU loss (N,5) vs (N,5): Smooth-L1 on centers + (1 - KFIoU).

    Returns per-sample loss of shape (N,).
    """
    if pred.numel() == 0:
        return pred.new_zeros((0,))
    xy_p, xy_t = pred[:, :2], target[:, :2]
    diff = (xy_p - xy_t).abs()
    xy_loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta).sum(-1)
    _, sig_p = _xywhr_to_sigma(pred)
    _, sig_t = _xywhr_to_sigma(target)
    vb_p = (4.0 * _det2(sig_p).clamp(min=0.0).sqrt()).clamp(min=eps)
    vb_t = (4.0 * _det2(sig_t).clamp(min=0.0).sqrt()).clamp(min=eps)
    kalman = torch.matmul(sig_p, _inv2(sig_p + sig_t))
    sig_kf = sig_p - torch.matmul(kalman, sig_p)
    vb = torch.nan_to_num(4.0 * _det2(sig_kf).clamp(min=0.0).sqrt(), nan=0.0)
    overlap = vb / (vb_p + vb_t - vb + eps)
    return xy_loss + (1.0 - overlap)


def rbox_to_aabb_xyxy(xywhr):
    """(N,5) cx,cy,w,h,theta -> (N,4) axis-aligned hull."""
    cx, cy, w, h, theta = xywhr.unbind(-1)
    c, s = torch.cos(theta).abs(), torch.sin(theta).abs()
    hw = (c * w + s * h) * 0.5
    hh = (s * w + c * h) * 0.5
    return torch.stack((cx - hw, cy - hh, cx + hw, cy + hh), dim=-1)
