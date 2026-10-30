"""One RTSP camera -> NV12 frames, decoded by GStreamer.

decoder="nv": rtspsrc ! depay ! h264parse ! nvh264dec (NVDEC hardware) ! cudadownload ! NV12 appsink
decoder="sw": ... ! avdec_h264 (1 thread, CPU) ! videoconvert ! NV12 appsink

Each stream runs a thread that keeps only the NEWEST decoded frame (a real camera pipeline drops
frames when inference is slower than the source; it does not build a queue and grow latency).
Frame timestamps (monotonic seconds):
  t_arrive  - the encoded access unit reached the pipeline (pad probe before the decoder)
  t_decoded - the decoded NV12 frame was pulled from the appsink
"""
import threading, time
from dataclasses import dataclass

import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

Gst.init(None)

PIPE = ("rtspsrc location={url} latency=0 protocols=tcp ! rtph264depay ! h264parse name=parse ! {dec} ! "
        "video/x-raw,format=NV12 ! appsink name=sink drop=true max-buffers=1 sync=false")
DECODERS = {"nv": "nvh264dec ! cudadownload", "sw": "avdec_h264 max-threads=1 ! videoconvert"}


@dataclass
class Frame:
    stream: int
    seq: int          # decoded-frame counter of this stream
    nv12: np.ndarray  # (h*3/2, w) uint8
    w: int
    h: int
    t_arrive: float
    t_decoded: float


class RtspSource:
    def __init__(self, stream_id, url, decoder="nv"):
        self.id, self.url = stream_id, url
        self.pipe = Gst.parse_launch(PIPE.format(url=url, dec=DECODERS[decoder]))
        self.sink = self.pipe.get_by_name("sink")
        self._arrive = {}            # pts -> arrival time, bounded
        self._lock = threading.Lock()
        self.latest = None           # newest undelivered Frame
        self.n_decoded = 0
        self.n_overwritten = 0       # decoded frames replaced before the consumer took them
        self.n_arrived = 0
        self.on_frame = lambda: None
        self._stop = threading.Event()
        self.pipe.get_by_name("parse").get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, self._probe)

    def _probe(self, pad, info):
        now = time.monotonic()
        buf = info.get_buffer()
        self.n_arrived += 1
        if buf.pts != Gst.CLOCK_TIME_NONE:
            self._arrive[buf.pts] = now
            if len(self._arrive) > 64:
                self._arrive.pop(next(iter(self._arrive)))
        return Gst.PadProbeReturn.OK

    def start(self):
        self.pipe.set_state(Gst.State.PLAYING)
        self.thread = threading.Thread(target=self._pull, daemon=True)
        self.thread.start()

    def _pull(self):
        while not self._stop.is_set():
            sample = self.sink.emit("try-pull-sample", Gst.SECOND // 4)
            if sample is None:
                continue
            t_dec = time.monotonic()
            s = sample.get_caps().get_structure(0)
            w, h = s.get_value("width"), s.get_value("height")
            buf = sample.get_buffer()
            ok, mi = buf.map(Gst.MapFlags.READ)
            if not ok:
                continue
            try:
                if mi.size != w * h * 3 // 2:  # padded strides are not handled; fail loudly
                    raise RuntimeError(f"unexpected NV12 buffer size {mi.size} for {w}x{h}")
                nv12 = np.frombuffer(mi.data, np.uint8).reshape(h * 3 // 2, w).copy()
            finally:
                buf.unmap(mi)
            t_arr = self._arrive.pop(buf.pts, t_dec)
            with self._lock:
                if self.latest is not None:
                    self.n_overwritten += 1
                self.latest = Frame(self.id, self.n_decoded, nv12, w, h, t_arr, t_dec)
                self.n_decoded += 1
            self.on_frame()

    def take(self):
        with self._lock:
            f, self.latest = self.latest, None
        return f

    def stop(self):
        self._stop.set()
        self.pipe.set_state(Gst.State.NULL)
