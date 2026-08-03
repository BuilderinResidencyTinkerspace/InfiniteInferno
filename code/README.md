# Agni vision controller

Agni is a Raspberry Pi 5–oriented, CPU-only vision pipeline for a people-following
prototype. It detects people with the included YOLOv8n ONNX model, assigns stable
session-local track IDs, selects one target, and emits bounded **intent commands**.
It does not connect to or arm a flight controller by itself.

## What works

- person-only YOLO inference through ONNX Runtime
- correct letterbox coordinate mapping and NumPy NMS
- multi-person tracking with stable IDs through brief missed detections
- target locking and proportional follow commands (`yaw`, `forward`, `vertical`)
- loss-of-target hover failsafe
- configurable gesture-to-action mapping
- optional COCO-pose gesture recognition when a YOLOv8 pose ONNX model is supplied
- Raspberry Pi Camera Module (`Picamera2`) and USB/video-file inputs
- configurable frame skipping (`model.inference_stride`) to reduce Pi CPU load
- deterministic dry-run mode and hardware-free unit tests

Track IDs identify a person only within the current process. They are not face
recognition or a permanent identity, and may change after a long occlusion.

## Raspberry Pi 5 / Raspberry Pi OS 64-bit

Use current 64-bit Raspberry Pi OS (Bookworm or newer). The camera and OpenCV are
best installed from Raspberry Pi OS packages; create the virtual environment with
system packages visible:

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv python3-venv
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m scripts.pi_smoke_test
```

Run without sending commands to a drone:

```bash
python -m src.main --config configs/config.yaml --display --auto-follow
```

For a USB camera use `--camera opencv --source 0`. For a video file use
`--camera opencv --source path/to/video.mp4`. Press `q` to stop the display.
`--auto-follow` locks the largest first visible person; for deliberate selection,
omit it and use `--target-id N` after observing the printed session track IDs.

## Gesture model

The included `models/yolov8n.onnx` is a detection model, so it can follow people
but has no body keypoints. To enable hand/arm gestures, export a small COCO pose
model with a fixed square input (for example a nano YOLO pose model), place it in
`models/`, and set `model.path` in `configs/config.yaml`. Agni automatically
recognizes pose output and currently supports:

```bash
# Run this on a development computer; ultralytics/torch are not Pi runtime deps.
python -m pip install ultralytics
yolo export model=yolov8n-pose.pt format=onnx imgsz=320 opset=12 simplify=True
```

- `right_hand_up`
- `left_hand_up`
- `both_hands_up`

Mappings are configured under `gestures.actions`. Actions are deliberately
limited to `start_follow`, `stop_follow`, `hover`, and `land`. A gesture must be
held for multiple frames and has a cooldown to reduce accidental activation.

## Safety boundary

The program prints JSON command intents. Integrate a MAVSDK/PX4/ArduPilot adapter
only after bench tests, a physical kill switch, geofence, low-battery handling,
and autopilot-native return-to-home/failsafe configuration exist. Keep propellers
removed during initial integration. Vision loss always produces hover here; the
autopilot must remain the final safety authority.

## Layout

- `src/detector.py` – ONNX Runtime session
- `src/preprocess.py` / `src/postprocess.py` – image transform and YOLO decoding
- `src/tracker.py` – session-local person IDs
- `src/gestures.py` – pose gesture state machine
- `src/controller.py` – safe, bounded follow intents
- `src/main.py` – CLI pipeline
- `tests/` – hardware-free regression suite
