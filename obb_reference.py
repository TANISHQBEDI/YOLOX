"""
Reference OBB (Oriented Bounding Box) additions for a YOLOX-style decoupled head.

This is NOT copied from Ultralytics or any AGPL codebase — it's original code written
to plug into YOLOX's (Apache-2.0) existing head structure. Adapt variable names /
integration points to match your actual forked repo's YOLOXHead class.

Angle representation: we predict (sin(2*theta), cos(2*theta)) instead of raw theta.
This avoids the 0/180-degree wraparound discontinuity that breaks naive regression
(a box at 179 degrees and 1 degree are visually almost identical, but raw regression
sees them as maximally far apart -> unstable gradients). Using the doubled angle also
naturally handles the fact that an oriented box has 180-degree symmetry (a box rotated
180 degrees looks identical to the original box).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. HEAD ADDITION
# ---------------------------------------------------------------------------
# In YOLOX's YOLOXHead.__init__, there are existing per-scale branches like:
#   self.cls_preds, self.reg_preds, self.obj_preds
# each built as nn.ModuleList over the FPN levels (usually 3 scales).
#
# Add a parallel branch for angle, built the same way the existing reg branch is
# (same input channels, same conv-stem pattern), but outputting 2 channels
# (sin2theta, cos2theta) per anchor point instead of 4 (x,y,w,h).

class AngleHead(nn.Module):
    """
    One angle-prediction branch for a single FPN scale.
    Mirrors the structure of YOLOX's existing reg_preds conv (same in_channels
    as whatever feature width that scale's stem produces).
    """

    def __init__(self, in_channels: int):
        super().__init__()
        # 2 output channels: sin(2*theta), cos(2*theta)
        self.angle_pred = nn.Conv2d(in_channels, 2, kernel_size=1, stride=1, padding=0)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.angle_pred(feat)  # (B, 2, H, W)


# In YOLOXHead.forward(), wherever reg_output = self.reg_preds[k](reg_feat) is computed,
# add right after it:
#
#   angle_output = self.angle_preds[k](reg_feat)   # reuse the SAME reg_feat stem
#
# and concatenate it into the per-scale output alongside reg/obj/cls, e.g.:
#
#   output = torch.cat([reg_output, obj_output, cls_output, angle_output], dim=1)
#
# then update wherever the channel-splitting happens downstream (decode step) to
# also slice out these 2 extra channels. This is the main place bugs creep in —
# double check every place the code assumes a fixed channel count.


# ---------------------------------------------------------------------------
# 2. DECODING PREDICTED ANGLE BACK TO THETA
# ---------------------------------------------------------------------------

def decode_angle(sin2theta: torch.Tensor, cos2theta: torch.Tensor) -> torch.Tensor:
    """
    Recover theta in radians, range (-pi/2, pi/2], from predicted sin(2*theta)/cos(2*theta).
    Normalizing first keeps atan2 numerically stable even if the raw predictions
    aren't perfectly unit-norm (they won't be, since they're just conv outputs, not
    explicitly normalized during training).
    """
    norm = torch.sqrt(sin2theta ** 2 + cos2theta ** 2).clamp(min=1e-6)
    sin2theta_n = sin2theta / norm
    cos2theta_n = cos2theta / norm
    two_theta = torch.atan2(sin2theta_n, cos2theta_n)  # range (-pi, pi]
    theta = two_theta / 2.0  # range (-pi/2, pi/2]
    return theta


def encode_angle(theta: torch.Tensor) -> torch.Tensor:
    """Ground-truth theta (radians) -> (sin(2*theta), cos(2*theta)) target."""
    two_theta = 2.0 * theta
    return torch.sin(two_theta), torch.cos(two_theta)


# ---------------------------------------------------------------------------
# 3. ANGLE LOSS
# ---------------------------------------------------------------------------
# Simple, stable starting point: smooth-L1 on the (sin2theta, cos2theta) pair
# directly. This is easier to debug than a full rotated-IoU loss and is a fine
# baseline to confirm the whole pipeline (head -> loss -> backward) is wired
# correctly before you invest in a fancier rotated-IoU or KLD loss.

def angle_loss(pred_sin2t: torch.Tensor, pred_cos2t: torch.Tensor,
                target_theta: torch.Tensor) -> torch.Tensor:
    """
    pred_sin2t, pred_cos2t: (N,) raw predictions for matched positive anchors
    target_theta: (N,) ground truth angle in radians for the same anchors
    """
    target_sin2t, target_cos2t = encode_angle(target_theta)
    loss_sin = F.smooth_l1_loss(pred_sin2t, target_sin2t, reduction="none")
    loss_cos = F.smooth_l1_loss(pred_cos2t, target_cos2t, reduction="none")
    return (loss_sin + loss_cos).mean()


# NOTE: once the basic pipeline trains and you want tighter localization,
# swap this for a rotated-IoU-based loss (e.g. via a differentiable rotated
# IoU implementation) — that directly optimizes what you actually care about
# (box overlap), rather than an angle-representation proxy. Get the simple
# version working end-to-end first.


# ---------------------------------------------------------------------------
# 4. ROTATION-SAFE AUGMENTATION (the part most OBB pipelines get wrong)
# ---------------------------------------------------------------------------
# When you rotate or flip an image during augmentation, the box's (x, y, w, h)
# AND theta must all be transformed together. Getting this wrong silently
# corrupts your labels and the model will train on garbage without erroring out.

def rotate_obb_label(cx, cy, w, h, theta, angle_deg, img_w, img_h):
    """
    Rotate a single OBB label by angle_deg (counter-clockwise, degrees) around
    the image center, matching an image rotation augmentation of the same angle.

    cx, cy: box center in pixels
    w, h: box width/height in pixels (unchanged by rotation)
    theta: box angle in radians
    angle_deg: the augmentation's rotation angle in degrees
    img_w, img_h: image dimensions (rotation pivot is the image center)

    Returns: new_cx, new_cy, w, h, new_theta
    """
    angle_rad = math.radians(angle_deg)
    icx, icy = img_w / 2.0, img_h / 2.0

    # rotate the center point around the image center
    dx, dy = cx - icx, cy - icy
    new_dx = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
    new_dy = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
    new_cx, new_cy = icx + new_dx, icy + new_dy

    # box orientation rotates by the same amount; wrap back into (-pi/2, pi/2]
    new_theta = theta + angle_rad
    new_theta = (new_theta + math.pi / 2) % math.pi - math.pi / 2

    return new_cx, new_cy, w, h, new_theta


def hflip_obb_label(cx, cy, w, h, theta, img_w):
    """Horizontal flip: mirror center x, negate theta."""
    new_cx = img_w - cx
    new_theta = -theta
    new_theta = (new_theta + math.pi / 2) % math.pi - math.pi / 2
    return new_cx, cy, w, h, new_theta


# ---------------------------------------------------------------------------
# 5. UNIT TESTS — run these with `pytest` on CPU before any GPU training
# ---------------------------------------------------------------------------

def test_encode_decode_roundtrip():
    """
    A predicted angle, encoded then decoded, should return the original angle.
    Note: +90 and -90 degrees represent the SAME physical box orientation under
    this 180-degree-symmetric encoding (a box rotated 180 degrees is identical
    to itself), so at that exact boundary either sign is a correct answer —
    we check equivalence mod 180 instead of exact equality there.
    """
    for theta_deg in [-89, -45, 0, 1, 45, 89]:
        theta = torch.tensor(math.radians(theta_deg))
        s, c = encode_angle(theta)
        recovered = decode_angle(s, c)
        assert abs(math.degrees(recovered.item()) - theta_deg) < 1e-3, \
            f"roundtrip failed for {theta_deg} degrees, got {math.degrees(recovered.item())}"

    # boundary case: 90 degrees may come back as -90, which is equivalent
    theta = torch.tensor(math.radians(90))
    s, c = encode_angle(theta)
    recovered_deg = math.degrees(decode_angle(s, c).item())
    assert abs(recovered_deg - 90) < 1e-3 or abs(recovered_deg + 90) < 1e-3, \
        f"90-degree boundary case failed, got {recovered_deg}"


def test_90_degree_rotation_swaps_orientation():
    """A box at theta=0 (horizontal), rotated 90 degrees, should end up at theta=+-90."""
    cx, cy, w, h, theta = 100, 100, 50, 20, 0.0
    img_w, img_h = 200, 200
    new_cx, new_cy, new_w, new_h, new_theta = rotate_obb_label(
        cx, cy, w, h, theta, angle_deg=90, img_w=img_w, img_h=img_h
    )
    assert abs(math.degrees(new_theta)) - 90 < 1e-3 or abs(abs(math.degrees(new_theta)) - 90) < 1e-3


def test_hflip_negates_angle():
    cx, cy, w, h, theta = 50, 100, 40, 15, math.radians(30)
    img_w = 200
    new_cx, new_cy, new_w, new_h, new_theta = hflip_obb_label(cx, cy, w, h, theta, img_w)
    assert new_cx == img_w - cx
    assert abs(math.degrees(new_theta) - (-30)) < 1e-3


if __name__ == "__main__":
    # quick manual run without pytest, for a fast sanity check
    test_encode_decode_roundtrip()
    test_90_degree_rotation_swaps_orientation()
    test_hflip_negates_angle()
    print("All sanity checks passed.")