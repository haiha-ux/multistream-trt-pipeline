"""YOLO11n .pt -> ONNX (dynamic batch) -> TensorRT FP16 engine with a batch profile 1..MAX_BATCH.

Adapted from yolo-tensorrt-edge/export.py. Writes a plain TensorRT plan (no metadata header).
"""
import argparse, subprocess, sys
from pathlib import Path

import tensorrt as trt

ROOT = Path(__file__).parent


def export_onnx(pt, imgsz):
    from ultralytics import YOLO
    return Path(YOLO(pt).export(format="onnx", imgsz=imgsz, opset=17, dynamic=True, simplify=False))


def build(onnx, out, imgsz, max_batch):
    log = trt.Logger(trt.Logger.WARNING)
    b = trt.Builder(log)
    net = b.create_network()
    if not trt.OnnxParser(net, log).parse(onnx.read_bytes()):
        raise SystemExit("ONNX parse failed")
    cfg = b.create_builder_config()
    cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    prof = b.create_optimization_profile()
    name = net.get_input(0).name
    shape = lambda n: (n, 3, imgsz, imgsz)
    prof.set_shape(name, shape(1), shape(max(1, max_batch // 2)), shape(max_batch))
    cfg.add_optimization_profile(prof)
    plan = b.build_serialized_network(net, cfg)
    if not plan:
        raise SystemExit("engine build failed")
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(bytes(plan))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "weights/yolo11n.pt"))
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default=str(ROOT / "engines/yolo11n_fp16_dyn.plan"))
    ap.add_argument("--onnx-only", action="store_true")
    a = ap.parse_args()
    onnx = Path(a.weights).with_suffix(".onnx")
    if "--onnx-only" in sys.argv:
        export_onnx(a.weights, a.imgsz)
        raise SystemExit
    # separate process: Ultralytics' CPU ONNX export hides the GPU from the rest of that process
    subprocess.run([sys.executable, __file__, "--onnx-only", "--weights", a.weights, "--imgsz", str(a.imgsz)], check=True)
    build(onnx, Path(a.out), a.imgsz, a.max_batch)
    print("engine:", a.out)
