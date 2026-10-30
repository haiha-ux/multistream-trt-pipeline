"""results/bench.json -> results/bench.png (left: throughput, right: stage latency stacked + end-to-end)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).parent / "results"
data = json.loads((R / "bench.json").read_text())
nv = [r for r in data["runs"] if r["decoder"] == "nv"]
ns = [r["streams"] for r in nv]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ns, [r["aggregate_fps"] for r in nv], "o-", label="processed (aggregate)")
ax[0].plot(ns, [n * data["source_fps"] for n in ns], "k--", lw=1, label="cameras deliver (30 fps each)")
ax[0].set(xlabel="RTSP streams", ylabel="frames/s", title="Throughput", xticks=ns)
ax[0].legend()
stages = [("decode", "decode (RTP in -> NV12 out)"), ("queue_wait", "wait for batch"),
          ("preprocess_per_batch", "preprocess"), ("infer_per_batch", "TensorRT infer"),
          ("postprocess_nms_per_batch", "NMS"), ("track_per_batch", "track")]
bottom = [0] * len(nv)
for k, label in stages:
    v = [r["latency_ms"][k]["median"] for r in nv]
    ax[1].bar(range(len(nv)), v, bottom=bottom, label=label)
    bottom = [b + x for b, x in zip(bottom, v)]
ax[1].plot(range(len(nv)), [r["latency_ms"]["end_to_end"]["p95"] for r in nv], "r^--", label="end-to-end p95")
ax[1].plot(range(len(nv)), [r["latency_ms"]["end_to_end"]["median"] for r in nv], "ko", label="end-to-end p50")
ax[1].set(xticks=range(len(nv)), xticklabels=ns, xlabel="RTSP streams", ylabel="ms (median per stage)", title="Latency per frame")
ax[1].legend(fontsize=7)
fig.suptitle("Shared NVIDIA GB10 (not a Jetson), NVDEC + TensorRT FP16 YOLO11n 640 + ByteTrack", fontsize=9)
fig.tight_layout()
fig.savefig(R / "bench.png", dpi=130)
