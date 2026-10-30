#!/usr/bin/env bash
# Fetch mediamtx, encode 4 MOT17 sequences to 1080p30 H.264 clips, build the TensorRT engine.
# Expects MOT17 train frames (CC BY-NC-SA 3.0) at $MOT17, e.g. from https://motchallenge.net/data/MOT17/
set -euo pipefail
cd "$(dirname "$0")/.."
MOT17=${MOT17:?set MOT17 to the MOT17/train directory}
mkdir -p bin data/clips weights engines
if [ ! -x bin/mediamtx ]; then
  curl -sL https://github.com/bluenviron/mediamtx/releases/download/v1.21.3/mediamtx_v1.21.3_linux_arm64.tar.gz | tar xz -C bin mediamtx
fi
for s in 02 04 09 10; do
  nice -n 19 ffmpeg -hide_banner -loglevel error -y -framerate 30 -i "$MOT17/MOT17-$s/img1/%06d.jpg" \
    -c:v libx264 -threads 2 -preset veryfast -profile:v main -g 30 -bf 0 -crf 23 -pix_fmt yuv420p -r 30 "data/clips/mot17_$s.mp4"
done
[ -f weights/yolo11n.pt ] || curl -L -o weights/yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
nice -n 19 python build_engine.py
