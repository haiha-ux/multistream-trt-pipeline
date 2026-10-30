"""Serve local video files as N RTSP streams: mediamtx + one `ffmpeg -re -c copy` publisher per stream.

No re-encoding at publish time (the clips are pre-encoded H.264), so the publishers cost almost no CPU.
Streams are rtsp://127.0.0.1:8554/cam0 ... camN-1. Clips are cycled and looped forever.
"""
import argparse, subprocess, tempfile, time
from pathlib import Path

ROOT = Path(__file__).parent
PORT = 8554
CFG = """logLevel: warn
rtsp: yes
rtspAddress: :{port}
rtspTransports: [tcp, udp]
rtmp: no
hls: no
webrtc: no
srt: no
api: no
metrics: no
pprof: no
playback: no
paths:
  all_others:
"""


def url(i):
    return f"rtsp://127.0.0.1:{PORT}/cam{i}"


class RtspFarm:
    """Context manager: start the server and n publishers, kill them on exit."""

    def __init__(self, n, clips=None):
        self.n = n
        self.clips = sorted(clips or (ROOT / "data/clips").glob("*.mp4"))
        if not self.clips:
            raise SystemExit("no clips in data/clips (see README: 'Prepare the clips')")
        self.procs = []

    def __enter__(self):
        cfg = Path(tempfile.mkdtemp()) / "mediamtx.yml"
        cfg.write_text(CFG.format(port=PORT))
        # mediamtx writes auto.crt/auto.key into its cwd, so run it in the temp dir
        self.procs.append(subprocess.Popen([str(ROOT / "bin/mediamtx"), str(cfg)], cwd=cfg.parent))
        time.sleep(1.0)
        for i in range(self.n):
            clip = self.clips[i % len(self.clips)]
            offset = 5 * (i // len(self.clips))  # repeat clips start at different points
            cmd = ["nice", "-n", "10", "ffmpeg", "-hide_banner", "-loglevel", "error", "-re", "-stream_loop", "-1",
                   "-ss", str(offset), "-i", str(clip), "-c", "copy", "-f", "rtsp", "-rtsp_transport", "tcp", url(i)]
            self.procs.append(subprocess.Popen(cmd))
        time.sleep(2.0)  # let publishers register
        return self

    def __exit__(self, *exc):
        for p in reversed(self.procs):
            p.terminate()
        for p in self.procs:
            p.wait()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--streams", type=int, default=4)
    a = ap.parse_args()
    with RtspFarm(a.streams):
        print("serving:", *(url(i) for i in range(a.streams)), sep="\n  ")
        while True:
            time.sleep(3600)
