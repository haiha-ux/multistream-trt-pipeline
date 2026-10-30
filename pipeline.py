"""N RTSP streams -> GStreamer decode -> batched TensorRT detector -> per-stream ByteTrack.

    python pipeline.py --streams 4 --seconds 30 --out results/run.json
    python pipeline.py --urls rtsp://cam1/... rtsp://cam2/... --preview preview.mp4

Per iteration the loop takes the newest frame of every stream that has one (up to the engine's max
batch), so batch size adapts to the load. Stage times are measured with a CUDA sync after each
stage, which costs a little throughput but makes the stage numbers honest.
"""
import argparse, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psutil
import torch

from engine import TrtEngine, decode_nms, preprocess_nv12
from rtsp_server import RtspFarm, url
from source import RtspSource

ROOT = Path(__file__).parent


def make_tracker(fps=30):
    from ultralytics.cfg import IterableSimpleNamespace
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import YAML
    import ultralytics
    cfg = YAML.load(Path(ultralytics.__file__).parent / "cfg/trackers/bytetrack.yaml")
    return BYTETracker(IterableSimpleNamespace(**cfg), frame_rate=fps)


def track_update(tracker, det, shape):
    from ultralytics.engine.results import Boxes
    return tracker.update(Boxes(det, shape).cpu().numpy())


class GpuSampler:
    """Polls nvidia-smi utilisation. On a shared GPU this includes other processes' work."""

    def __init__(self, period=0.5):
        import subprocess
        self.vals = []
        self.p = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
                                   "-lms", str(int(period * 1000))], stdout=subprocess.PIPE, text=True)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            try:
                self.vals.append(float(line))
            except ValueError:
                pass

    def stop(self):
        self.p.terminate()
        return self.vals


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


def summarize(a):
    return {"median": pct(a, 50), "p95": pct(a, 95), "n": len(a)}


def run(urls, seconds, decoder="nv", engine_path=None, conf=0.25, warmup=6.0, preview=None, preview_streams=4,
        log=print):
    n = len(urls)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.3)
    eng = TrtEngine(engine_path or ROOT / "engines/yolo11n_fp16_dyn.plan")
    S = eng.size
    trackers = [make_tracker() for _ in range(n)]
    srcs = [RtspSource(i, u, decoder) for i, u in enumerate(urls)]
    wake = threading.Event()
    for s in srcs:
        s.on_frame = wake.set
    # warm up TensorRT/CUDA for every batch size we may see
    for b in sorted({1, 2, 4, 8, min(n, eng.max_batch)}):
        if b <= eng.max_batch:
            for _ in range(3):
                eng.infer(b)
    torch.cuda.synchronize()

    pv = Preview(preview, preview_streams) if preview else None
    rec = {"decode": [], "wait": [], "e2e": [], "frame_stream": []}
    bat = {"size": [], "pre": [], "infer": [], "post": [], "track": [], "total": [], "track_per_frame": []}
    proc = psutil.Process()
    for s in srcs:
        s.start()
    t_start = time.monotonic()
    t_meas = None
    gpu = None
    counts0 = None
    scratch = torch.empty(eng.max_batch, 3, S, S, device="cuda")
    pinned = {}
    jobs = ThreadPoolExecutor(max_workers=1)
    pending = None

    def finish(frames, dets, t0, t1, t2, t3, measuring):
        tt = []
        for f, d in zip(frames, dets):
            ta = time.monotonic()
            tracks = track_update(trackers[f.stream], d, (f.h, f.w))
            tt.append(time.monotonic() - ta)
            if pv:
                pv.add(f, tracks)
        t4 = time.monotonic()
        if not measuring:
            return
        bat["size"].append(len(frames))
        bat["pre"].append((t1 - t0) * 1e3)
        bat["infer"].append((t2 - t1) * 1e3)
        bat["post"].append((t3 - t2) * 1e3)
        bat["track"].append((t4 - t3) * 1e3)  # includes any wait for the worker thread
        bat["total"].append((t4 - t0) * 1e3)
        bat["track_per_frame"] += [x * 1e3 for x in tt]
        for f in frames:
            rec["decode"].append((f.t_decoded - f.t_arrive) * 1e3)
            rec["wait"].append((t0 - f.t_decoded) * 1e3)
            rec["e2e"].append((t4 - f.t_arrive) * 1e3)
            rec["frame_stream"].append(f.stream)

    while True:
        now = time.monotonic()
        if t_meas is None and now - t_start >= warmup:
            t_meas = now
            gpu = GpuSampler()
            proc.cpu_percent(None)
            psutil.cpu_percent(None)
            counts0 = [(s.n_arrived, s.n_decoded, s.n_overwritten) for s in srcs]
            for d in (rec, bat):
                for k in d:
                    d[k].clear()
        if t_meas is not None and now - t_meas >= seconds:
            break
        wake.wait(0.1)
        wake.clear()
        frames = [f for f in (s.take() for s in srcs) if f is not None][:eng.max_batch]
        if not frames:
            continue
        B = len(frames)
        t0 = time.monotonic()
        # ---- preprocess: H2D copy of NV12 + resize + colour convert + letterbox, per frame into the batch tensor
        metas = [None] * B
        groups = {}
        for i, f in enumerate(frames):
            groups.setdefault((f.h, f.w), []).append(i)
        for (h, w), idx in groups.items():  # same-resolution frames go through one set of kernels
            if (h, w) not in pinned:
                pinned[(h, w)] = torch.empty(eng.max_batch, h * 3 // 2, w, dtype=torch.uint8).pin_memory()
            host = pinned[(h, w)]
            for j, i in enumerate(idx):
                host[j].copy_(torch.from_numpy(frames[i].nv12))
            r, left, top = preprocess_nv12(host[:len(idx)].cuda(non_blocking=True), h, w, S, scratch[:len(idx)])
            eng.x[torch.tensor(idx, device="cuda")] = scratch[:len(idx)].to(eng.x.dtype)
            for i in idx:
                metas[i] = (r, left, top)
        torch.cuda.synchronize()
        t1 = time.monotonic()
        raw = eng.infer(B)
        torch.cuda.synchronize()
        t2 = time.monotonic()
        dets = decode_nms(raw, metas, conf_thres=conf)
        t3 = time.monotonic()
        # ---- track (CPU) runs on a worker thread so it overlaps the next batch's GPU work;
        # one worker => batches (and so each stream's frames) are tracked in order
        if pending is not None:
            pending.result()  # at most one batch in flight: a slow tracker shows up as latency, not as a hidden queue
        pending = jobs.submit(finish, frames, dets, t0, t1, t2, t3, t_meas is not None)
    dur = time.monotonic() - t_meas
    jobs.shutdown(wait=True)
    gpu_vals = gpu.stop()
    cpu_proc = proc.cpu_percent(None) / 100  # cores used by this process (decode threads included)
    cpu_sys = psutil.cpu_percent(None)
    c1 = [(s.n_arrived, s.n_decoded, s.n_overwritten) for s in srcs]
    for s in srcs:
        s.stop()
    if pv:
        pv.close()
    arrived = sum(b[0] - a[0] for a, b in zip(counts0, c1))
    decoded = sum(b[1] - a[1] for a, b in zip(counts0, c1))
    processed = len(rec["e2e"])
    per_stream = [rec["frame_stream"].count(i) / dur for i in range(n)]
    return {
        "streams": n, "decoder": decoder, "seconds": round(dur, 1), "engine_max_batch": eng.max_batch,
        "aggregate_fps": processed / dur, "per_stream_fps_min": min(per_stream), "per_stream_fps_mean": float(np.mean(per_stream)),
        "arrived_fps": arrived / dur, "decoded_fps": decoded / dur,
        "dropped_pct": 100 * (1 - processed / max(arrived, 1)),
        "batch_size_mean": float(np.mean(bat["size"])), "batches_per_s": len(bat["size"]) / dur,
        "latency_ms": {
            "decode": summarize(rec["decode"]), "queue_wait": summarize(rec["wait"]),
            "preprocess_per_batch": summarize(bat["pre"]), "infer_per_batch": summarize(bat["infer"]),
            "postprocess_nms_per_batch": summarize(bat["post"]), "track_per_batch": summarize(bat["track"]),
            "track_per_frame": summarize(bat["track_per_frame"]), "batch_total": summarize(bat["total"]),
            "end_to_end": summarize(rec["e2e"]),
        },
        "gpu_util_pct": {"mean": float(np.mean(gpu_vals)) if gpu_vals else None, "max": max(gpu_vals) if gpu_vals else None},
        "cpu_cores_process": cpu_proc, "cpu_util_system_pct": cpu_sys,
    }


class Preview:
    """Writes a mosaic of the first k streams (boxes + track ids, head regions blurred, low-res)."""

    TILE = (320, 180)

    def __init__(self, path, k, fps=15):
        import cv2
        self.cv2, self.k, self.fps = cv2, k, fps
        cols = 2 if k > 1 else 1
        self.cols, self.rows = cols, (k + cols - 1) // cols
        self.tiles = [np.zeros((self.TILE[1], self.TILE[0], 3), np.uint8) for _ in range(k)]
        self.w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (self.TILE[0] * cols, self.TILE[1] * self.rows))
        self.next = time.monotonic()

    def add(self, f, tracks):
        if f.stream >= self.k:
            return
        cv2 = self.cv2
        img = cv2.cvtColor(f.nv12, cv2.COLOR_YUV2BGR_NV12)
        for x1, y1, x2, y2, tid, *_ in tracks:  # blur the top 22% of every person box (faces) before drawing
            x1, y1, x2, y2 = max(int(x1), 0), max(int(y1), 0), int(x2), int(y2)
            hh = max(int((y2 - y1) * 0.22), 2)
            roi = img[y1:y1 + hh, x1:x2]
            if roi.size:
                img[y1:y1 + hh, x1:x2] = cv2.GaussianBlur(roi, (0, 0), max(hh / 3, 3))
        for x1, y1, x2, y2, tid, *_ in tracks:
            col = [(int(tid) * 67) % 200 + 55, (int(tid) * 131) % 200 + 55, (int(tid) * 29) % 200 + 55]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 3)
            cv2.putText(img, str(int(tid)), (int(x1), max(int(y1) - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
        self.tiles[f.stream] = cv2.resize(img, self.TILE, interpolation=cv2.INTER_AREA)
        now = time.monotonic()
        if now >= self.next:
            self.next = now + 1 / self.fps
            rows = [np.hstack(self.tiles[r * self.cols:(r + 1) * self.cols] + [np.zeros_like(self.tiles[0])] * (self.cols - len(self.tiles[r * self.cols:(r + 1) * self.cols]))) for r in range(self.rows)]
            self.w.write(np.vstack(rows))

    def close(self):
        self.w.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--streams", type=int, default=4, help="serve and read N local RTSP streams")
    ap.add_argument("--urls", nargs="+", help="use these RTSP URLs instead of the local farm")
    ap.add_argument("--decoder", choices=["nv", "sw"], default="nv", help="nv = NVDEC (nvh264dec), sw = libav on CPU")
    ap.add_argument("--engine", default=str(ROOT / "engines/yolo11n_fp16_dyn.plan"))
    ap.add_argument("--seconds", type=float, default=30, help="measured seconds (after a 6 s warm-up)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--preview", help="write an annotated mosaic mp4 of the first streams")
    ap.add_argument("--out", help="write the result JSON here")
    a = ap.parse_args()
    if a.urls:
        r = run(a.urls, a.seconds, a.decoder, a.engine, a.conf, preview=a.preview)
    else:
        with RtspFarm(a.streams):
            r = run([url(i) for i in range(a.streams)], a.seconds, a.decoder, a.engine, a.conf, preview=a.preview)
    print(json.dumps(r, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(exist_ok=True)
        Path(a.out).write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
