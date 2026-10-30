"""Batched GPU pre/post-processing and the TensorRT runner.

preprocess_nv12   NV12 frame -> letterboxed RGB float [0,1] 3xSxS, all on the GPU
decode_nms        raw YOLO output (B, 4+nc, A) -> per-image [x1,y1,x2,y2,conf,cls] in source pixels
TrtEngine         plain TensorRT 10 runner for a dynamic-batch engine (torch tensors as buffers)
"""
import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms

PAD = 114 / 255  # Ultralytics letterbox grey


def letterbox_params(h, w, size):
    """Scale and (left, top) padding used to fit an h x w image into size x size."""
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    return r, (size - nw) // 2, (size - nh) // 2, nh, nw


def nv12_to_rgb(y, uv):
    """BT.601 limited-range YUV -> RGB in [0,1]. y: (B,1,H,W), uv: (B,2,H,W) both float in [0,255]."""
    yy = (y - 16) * 1.164
    u, v = uv[:, 0:1] - 128, uv[:, 1:2] - 128
    rgb = torch.cat([yy + 1.596 * v, yy - 0.392 * u - 0.813 * v, yy + 2.017 * u], 1)
    return (rgb / 255).clamp_(0, 1)


def preprocess_nv12(nv12, h, w, size, out):
    """nv12: (B, h*3/2, w) uint8 CUDA tensor, B frames of the same resolution.
    Writes the letterboxed RGB images into out (B,3,size,size); returns (r, left, top).

    Resize happens in YUV space (the UV plane is half size) and the colour conversion runs on the
    already-small image, so the full 1080p frame is never expanded to RGB. One set of kernels per
    batch, not per frame: on a busy GPU the number of kernel launches is what costs.
    """
    B = nv12.shape[0]
    r, left, top, nh, nw = letterbox_params(h, w, size)
    y = nv12[:, :h].reshape(B, 1, h, w).float()
    uv = nv12[:, h:].reshape(B, h // 2, w // 2, 2).permute(0, 3, 1, 2).float()
    y = F.interpolate(y, (nh, nw), mode="bilinear", antialias=True)
    uv = F.interpolate(uv, (nh, nw), mode="bilinear", antialias=True)
    out.fill_(PAD)
    out[:, :, top:top + nh, left:left + nw] = nv12_to_rgb(y, uv)
    return r, left, top


def decode_nms(raw, metas, conf_thres=0.25, iou_thres=0.7, classes=(0,), max_det=300, pre_k=300):
    """raw: (B, 4+nc, A) fp32 (xywh in letterbox pixels + class scores). metas: [(r, left, top)] per image.

    GPU: best class score per anchor, top-k anchors per image, ONE device->host copy.
    CPU: confidence filter + class-aware NMS on at most pre_k boxes per image (sub-millisecond).
    Doing NMS on the host avoids several GPU sync round-trips, which are expensive when the GPU is shared.
    """
    B = raw.shape[0]
    scores = raw[:, 4:].transpose(1, 2)[..., list(classes)]      # (B, A, nc_sel)
    conf, cls_i = scores.max(-1)                                  # (B, A)
    top_c, idx = conf.topk(min(pre_k, conf.shape[1]), dim=1)      # (B, K)
    boxes = torch.gather(raw[:, :4].transpose(1, 2), 1, idx[..., None].expand(-1, -1, 4))
    cls_top = torch.gather(cls_i, 1, idx)
    packed = torch.cat([boxes, top_c[..., None], cls_top[..., None].float()], -1).cpu()  # (B, K, 6)
    cls_ids = torch.tensor(list(classes), dtype=torch.float32)
    out = []
    for i in range(B):
        d = packed[i]
        d = d[d[:, 4] > conf_thres]
        if len(d) == 0:
            out.append(np.zeros((0, 6), np.float32))
            continue
        xyxy = torch.stack([d[:, 0] - d[:, 2] / 2, d[:, 1] - d[:, 3] / 2, d[:, 0] + d[:, 2] / 2, d[:, 1] + d[:, 3] / 2], 1)
        keep = batched_nms(xyxy, d[:, 4], d[:, 5].long(), iou_thres)[:max_det]
        r, left, top = metas[i]
        xyxy = xyxy[keep]
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - left) / r
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - top) / r
        out.append(torch.cat([xyxy, d[keep, 4:5], cls_ids[d[keep, 5].long()][:, None]], 1).numpy())
    return out


class TrtEngine:
    """Dynamic-batch TensorRT engine, one execution context, buffers preallocated for max batch."""

    def __init__(self, plan_path):
        rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = rt.deserialize_cuda_engine(open(plan_path, "rb").read())
        self.ctx = self.engine.create_execution_context()
        self.inp, self.outp = self.engine.get_tensor_name(0), self.engine.get_tensor_name(1)
        self.max_batch = self.engine.get_tensor_profile_shape(self.inp, 0)[2][0]
        self.size = self.engine.get_tensor_shape(self.inp)[-1]
        out_shape = list(self.engine.get_tensor_shape(self.outp))
        dt = lambda n: torch.float16 if self.engine.get_tensor_dtype(n) == trt.float16 else torch.float32
        self.x = torch.zeros(self.max_batch, 3, self.size, self.size, device="cuda", dtype=dt(self.inp))
        out_shape[0] = self.max_batch
        self.y = torch.zeros(out_shape, device="cuda", dtype=dt(self.outp))

    def infer(self, n):
        """Run the first n images of self.x; returns a view of the first n outputs (fp32)."""
        self.ctx.set_input_shape(self.inp, (n, 3, self.size, self.size))
        self.ctx.set_tensor_address(self.inp, self.x.data_ptr())
        self.ctx.set_tensor_address(self.outp, self.y.data_ptr())
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return self.y[:n].float()
