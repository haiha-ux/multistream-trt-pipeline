"""CPU-only checks of the letterbox / colour / NMS maths (no GPU, no TensorRT engine needed)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from engine import PAD, decode_nms, letterbox_params, preprocess_nv12  # noqa: E402


def test_letterbox_1080p():
    r, left, top, nh, nw = letterbox_params(1080, 1920, 640)
    assert (nh, nw, left, top) == (360, 640, 0, 140)
    assert abs(r - 1 / 3) < 1e-9


def test_preprocess_white_frame_is_letterboxed():
    h, w = 1080, 1920
    nv12 = torch.cat([torch.full((h, w), 235, dtype=torch.uint8), torch.full((h // 2, w), 128, dtype=torch.uint8)])[None]
    out = torch.zeros(1, 3, 640, 640)
    r, left, top = preprocess_nv12(nv12, h, w, 640, out)
    assert (left, top) == (0, 140)
    assert torch.allclose(out[0, :, 0, :], torch.full((3, 640), PAD))       # padding rows
    assert torch.allclose(out[0, :, 320, 320], torch.ones(3), atol=0.01)    # video area: white (Y=235 -> 1.0)


def _raw(boxes_per_image, anchors=50, nc=80):
    """Build a fake (B, 4+nc, A) YOLO output; each box is (cx, cy, w, h, person_score)."""
    raw = torch.zeros(len(boxes_per_image), 4 + nc, anchors)
    for b, boxes in enumerate(boxes_per_image):
        for a, (cx, cy, bw, bh, sc) in enumerate(boxes):
            raw[b, :4, a] = torch.tensor([cx, cy, bw, bh])
            raw[b, 4, a] = sc
    return raw


def test_nms_suppresses_duplicates_per_image_and_maps_back_to_source_pixels():
    raw = _raw([
        [(320, 320, 100, 200, 0.9), (322, 322, 100, 200, 0.8)],   # duplicates -> keep the 0.9 one
        [(320, 320, 100, 200, 0.7)],                              # same box in image 1 must NOT be suppressed by image 0
        [],
    ])
    metas = [(1 / 3, 0, 140)] * 3
    dets = decode_nms(raw, metas)
    assert [len(d) for d in dets] == [1, 1, 0]
    x1, y1, x2, y2, conf, cls = dets[0][0]
    # (270, 220, 370, 420) in letterbox pixels -> x/r, (y-140)/r in the 1080p frame
    np.testing.assert_allclose([x1, y1, x2, y2], [810, 240, 1110, 840], atol=1e-3)
    assert abs(conf - 0.9) < 1e-6 and cls == 0
    assert abs(dets[1][0][4] - 0.7) < 1e-6


def test_nms_confidence_threshold():
    dets = decode_nms(_raw([[(100, 100, 50, 50, 0.2)]]), [(1.0, 0, 0)], conf_thres=0.25)
    assert len(dets[0]) == 0
