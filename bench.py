"""Sweep the number of RTSP streams and record per-stage latency, throughput and utilisation.

    python bench.py --streams 1 2 4 8 12 16 --seconds 30 --sw-decode-at 4 8

Starts one RTSP farm with max(streams) cameras, then runs the pipeline on the first N of them.
A run is "saturated" when it processes < 95% of the frames the cameras deliver; the sweep runs one
more step after the first saturated N, then stops. Output: results/bench.json, results/bench.csv.
"""
import argparse, csv, json, os, subprocess, time
from pathlib import Path

from pipeline import GpuSampler, run
from rtsp_server import RtspFarm, url

ROOT = Path(__file__).parent
FPS_SRC = 30


def wait_for_low_load(limit=30):
    while os.getloadavg()[0] > limit:
        print("load high, waiting", os.getloadavg()[0], flush=True)
        time.sleep(20)


def gpu_baseline(seconds=4):
    g = GpuSampler()
    time.sleep(seconds)
    v = g.stop()
    return sum(v) / len(v) if v else None


def flat(r):
    row = {"streams": r["streams"], "decoder": r["decoder"], "aggregate_fps": r["aggregate_fps"],
           "per_stream_fps_min": r["per_stream_fps_min"], "dropped_pct": r["dropped_pct"],
           "batch_size_mean": r["batch_size_mean"], "gpu_util_mean_pct": r["gpu_util_pct"]["mean"],
           "cpu_cores_process": r["cpu_cores_process"]}
    for k, v in r["latency_ms"].items():
        row[f"{k}_p50_ms"], row[f"{k}_p95_ms"] = v["median"], v["p95"]
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--streams", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16])
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--sw-decode-at", type=int, nargs="*", default=[4, 8], help="also run these N with CPU decode")
    a = ap.parse_args()
    out = {"gpu_util_before_pct": gpu_baseline(), "source_fps": FPS_SRC, "runs": []}
    print("GPU util with no pipeline running (other users):", out["gpu_util_before_pct"], flush=True)
    saturated_seen = False
    with RtspFarm(max(a.streams + a.sw_decode_at)) as farm:
        for n in a.streams:
            wait_for_low_load()
            r = run([url(i) for i in range(n)], a.seconds, "nv")
            out["runs"].append(r)
            print(f"N={n}: {r['aggregate_fps']:.0f} fps, e2e p50 {r['latency_ms']['end_to_end']['median']:.0f} ms, "
                  f"drop {r['dropped_pct']:.1f}%", flush=True)
            if saturated_seen:
                break
            saturated_seen = r["aggregate_fps"] < 0.95 * n * FPS_SRC
            time.sleep(3)
        for n in a.sw_decode_at:
            wait_for_low_load()
            r = run([url(i) for i in range(n)], a.seconds, "sw")
            out["runs"].append(r)
            print(f"N={n} sw-decode: {r['aggregate_fps']:.0f} fps, e2e p50 {r['latency_ms']['end_to_end']['median']:.0f} ms", flush=True)
            time.sleep(3)
    out["machine"] = {"gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                            capture_output=True, text=True).stdout.strip(), "cpu_cores": os.cpu_count()}
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    (res / "bench.json").write_text(json.dumps(out, indent=1))
    rows = [flat(r) for r in out["runs"]]
    with open(res / "bench.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
