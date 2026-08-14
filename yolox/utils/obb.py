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
    "canonicalize_xyxy_theta",
    "oriented_xyxy_to_corners",
    "transform_obb_xyxy",
    "kfiou_loss",
    "kfiou_matrix",
    "rbox_iou",
    "rbox_to_aabb_xyxy",
    "rbox_prior_loss",
    "rbox_prior_keep",
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


def rbox_prior_loss(pred_rbox, aspect_min=0.0, aspect_max=100.0, min_side=0.0, max_side=1.0e9):
    """
    Soft object-shape prior on decoded (cx, cy, w, h, θ).

    Returns per-row (aspect_loss, size_loss) of shape (N,).
    Weights of 0 in the head skip adding these to the total.
    """
    w = pred_rbox[:, 2].clamp(min=1e-3)
    h = pred_rbox[:, 3].clamp(min=1e-3)
    long = torch.maximum(w, h)
    short = torch.minimum(w, h)
    aspect = long / short
    a_min = pred_rbox.new_tensor(float(aspect_min))
    a_max = pred_rbox.new_tensor(float(aspect_max))
    s_min = pred_rbox.new_tensor(float(min_side))
    s_max = pred_rbox.new_tensor(float(max_side))
    loss_aspect = F.relu(aspect - a_max) + F.relu(a_min - aspect)
    loss_size = F.relu(s_min - short) + F.relu(long - s_max)
    return loss_aspect, loss_size


def rbox_prior_keep(pred_rbox, aspect_min=0.0, aspect_max=100.0, min_side=0.0, max_side=1.0e9):
    """Boolean mask of boxes that satisfy the object prior (inference filter)."""
    w = pred_rbox[:, 2].clamp(min=1e-3)
    h = pred_rbox[:, 3].clamp(min=1e-3)
    long = torch.maximum(w, h)
    short = torch.minimum(w, h)
    aspect = long / short
    return (
        (aspect >= float(aspect_min))
        & (aspect <= float(aspect_max))
        & (short >= float(min_side))
        & (long <= float(max_side))
    )


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
    cx_out, cy_out = np.asarray(cx), np.asarray(cy)
    w_arr = np.asarray(w, dtype=np.float64)
    h_arr = np.asarray(h, dtype=np.float64)
    th_arr = np.asarray(theta, dtype=np.float64)
    swap = h_arr > w_arr
    w_new = np.where(swap, h_arr, w_arr)
    h_new = np.where(swap, w_arr, h_arr)
    th_new = wrap_angle(np.where(swap, th_arr + math.pi / 2.0, th_arr))
    if np.ndim(w_new) == 0:
        return float(cx_out), float(cy_out), float(w_new), float(h_new), float(th_new)
    return (
        cx_out.astype(np.float32, copy=False),
        cy_out.astype(np.float32, copy=False),
        w_new.astype(np.float32),
        h_new.astype(np.float32),
        th_new.astype(np.float32),
    )


def canonicalize_xyxy_theta(xyxy, theta):
    """Canonicalize oriented extents stored as xyxy + theta (numpy)."""
    xyxy = np.asarray(xyxy, dtype=np.float32)
    if xyxy.size == 0:
        return xyxy.reshape(0, 4), np.asarray(theta, dtype=np.float32).reshape(-1)
    theta = np.asarray(theta, dtype=np.float32).reshape(-1)
    cx = (xyxy[:, 0] + xyxy[:, 2]) * 0.5
    cy = (xyxy[:, 1] + xyxy[:, 3]) * 0.5
    w = xyxy[:, 2] - xyxy[:, 0]
    h = xyxy[:, 3] - xyxy[:, 1]
    cx, cy, w, h, theta = canonicalize_rbox(cx, cy, w, h, theta)
    out = np.stack((cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5), axis=1)
    return out.astype(np.float32, copy=False), np.asarray(theta, dtype=np.float32).reshape(-1)


def oriented_xyxy_to_corners(xyxy, theta):
    """Oriented extents (N,4) + theta (N,) -> corners (N,4,2)."""
    xyxy = np.asarray(xyxy, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    cx = (xyxy[:, 0] + xyxy[:, 2]) * 0.5
    cy = (xyxy[:, 1] + xyxy[:, 3]) * 0.5
    hw = (xyxy[:, 2] - xyxy[:, 0]) * 0.5
    hh = (xyxy[:, 3] - xyxy[:, 1]) * 0.5
    c = np.cos(theta)
    s = np.sin(theta)
    dx = np.stack((hw, hw, -hw, -hw), axis=1)
    dy = np.stack((hh, -hh, -hh, hh), axis=1)
    rx = c[:, None] * dx - s[:, None] * dy
    ry = s[:, None] * dx + c[:, None] * dy
    return np.stack((cx[:, None] + rx, cy[:, None] + ry), axis=-1)


def _corners_to_canonical_rbox(corners):
    """Fit min-area rectangles to (N,4,2) corners and canonicalize."""
    n = corners.shape[0]
    cx = np.empty(n, dtype=np.float32)
    cy = np.empty(n, dtype=np.float32)
    w = np.empty(n, dtype=np.float32)
    h = np.empty(n, dtype=np.float32)
    theta = np.empty(n, dtype=np.float32)
    for i, pts in enumerate(corners):
        (rcx, rcy), (bw, bh), angle_deg = cv2.minAreaRect(pts.astype(np.float32))
        rcx, rcy, bw, bh, th = canonicalize_rbox(
            float(rcx), float(rcy), float(bw), float(bh),
            wrap_angle(math.radians(float(angle_deg))),
        )
        cx[i], cy[i], w[i], h[i], theta[i] = rcx, rcy, bw, bh, th
    return cx, cy, w, h, theta


def transform_obb_xyxy(targets, M, img_w, img_h):
    """
    Apply a 2x3 affine to labels [x1, y1, x2, y2, cls, theta].

    Corners go through the full matrix (rotation, isotropic scale, shear,
    translation). The parallelogram is refit with minAreaRect, then
    canonicalized so w >= h. Boxes whose center is outside the canvas
    or whose size collapses are dropped.

    Do not also multiply w,h by scale or add angle to theta: those are
    already in M. Clipping unrotated xyxy is wrong for a rotated box.
    """
    targets = ensure_theta_column(targets)
    if len(targets) == 0:
        return targets
    corners = oriented_xyxy_to_corners(targets[:, :4], targets[:, 5])
    ones = np.ones((corners.shape[0], 4, 1), dtype=np.float64)
    warped = np.concatenate((corners, ones), axis=-1) @ np.asarray(M, dtype=np.float64).T
    cx, cy, w, h, theta = _corners_to_canonical_rbox(warped)
    keep = (
        (cx >= 0) & (cx < float(img_w))
        & (cy >= 0) & (cy < float(img_h))
        & (w > 1.0) & (h > 1.0)
        & np.isfinite(theta)
    )
    if not np.any(keep):
        return np.zeros((0, targets.shape[1]), dtype=targets.dtype)
    out = targets[keep].copy()
    cx, cy, w, h, theta = cx[keep], cy[keep], w[keep], h[keep], theta[keep]
    out[:, 0] = cx - w * 0.5
    out[:, 1] = cy - h * 0.5
    out[:, 2] = cx + w * 0.5
    out[:, 3] = cy + h * 0.5
    out[:, 5] = theta
    return out


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


def rbox_iou(boxes_a, boxes_b):
    """
    Pairwise rotated IoU for matching.

    boxes_*: (N, 5) / (M, 5) as cx, cy, w, h, theta_radians.
    Prefers torchvision.ops.box_iou_rotated (same degrees convention as
    nms_rotated). Falls back to OpenCV rotatedRectangleIntersection so a
    wrong tilt is a low match even when that op is missing.
    """
    n, m = boxes_a.shape[0], boxes_b.shape[0]
    if n == 0 or m == 0:
        return boxes_a.new_zeros((n, m))

    try:
        import torchvision

        iou_fn = getattr(torchvision.ops, "box_iou_rotated", None)
    except Exception:
        iou_fn = None

    if iou_fn is not None:
        def _deg(boxes):
            return torch.stack(
                (
                    boxes[:, 0],
                    boxes[:, 1],
                    boxes[:, 2].clamp(min=1e-3),
                    boxes[:, 3].clamp(min=1e-3),
                    boxes[:, 4] * (180.0 / math.pi),
                ),
                dim=1,
            )

        iou = iou_fn(_deg(boxes_a), _deg(boxes_b))
        return torch.nan_to_num(iou, nan=0.0).clamp(0.0, 1.0)

    a = boxes_a.detach().to(dtype=torch.float32, device="cpu").numpy()
    b = boxes_b.detach().to(dtype=torch.float32, device="cpu").numpy()
    ious = _pairwise_rbox_iou_cv2(a, b)
    return torch.as_tensor(ious, device=boxes_a.device, dtype=boxes_a.dtype)


def _pairwise_rbox_iou_cv2(boxes_a, boxes_b):
    """numpy (N,5) (M,5) cx,cy,w,h,theta_rad -> (N,M) IoU."""
    ious = np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float64)
    a_deg = np.concatenate(
        (boxes_a[:, :4], boxes_a[:, 4:5] * (180.0 / math.pi)), axis=1
    )
    b_deg = np.concatenate(
        (boxes_b[:, :4], boxes_b[:, 4:5] * (180.0 / math.pi)), axis=1
    )
    for i, ra in enumerate(a_deg):
        r1 = ((float(ra[0]), float(ra[1])), (float(ra[2]), float(ra[3])), float(ra[4]))
        area_a = float(ra[2]) * float(ra[3])
        for j, rb in enumerate(b_deg):
            r2 = ((float(rb[0]), float(rb[1])), (float(rb[2]), float(rb[3])), float(rb[4]))
            ret, pts = cv2.rotatedRectangleIntersection(r1, r2)
            if ret == 0 or pts is None:
                continue
            inter = float(cv2.contourArea(pts))
            union = area_a + float(rb[2]) * float(rb[3]) - inter
            if union > 0:
                ious[i, j] = inter / union
    return ious
