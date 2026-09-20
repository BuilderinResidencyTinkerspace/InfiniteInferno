# ARGUS — Autonomous Vision-Based Drone

> A modular computer-vision and autonomous-drone development project combining real-time person detection, tracking, gesture intent recognition, bounded follow-control logic, Raspberry Pi/UNO Q experimentation, Pixhawk flight control, and a custom drone frame.

[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%205-red)](https://www.raspberrypi.com/products/raspberry-pi-5/)
[![Model](https://img.shields.io/badge/Model-YOLOv8n-blue)](https://github.com/ultralytics/ultralytics)
[![Inference](https://img.shields.io/badge/Inference-ONNX%20Runtime-green)](https://onnxruntime.ai/)
[![Flight Controller](https://img.shields.io/badge/Flight%20Controller-Pixhawk-orange)](https://pixhawk.org/)
[![Firmware](https://img.shields.io/badge/Firmware-PX4-purple)](https://px4.io/)
[![Language](https://img.shields.io/badge/Language-Python-yellow)](https://www.python.org/)

---

## Table of Contents

- [Overview](#overview)
- [Project Objective](#project-objective)
- [System Architecture](#system-architecture)
- [Current Software Implementation](#current-software-implementation)
- [Computer Vision Pipeline](#computer-vision-pipeline)
- [Person Detection](#person-detection)
- [Tracking](#tracking)
- [Gesture Recognition](#gesture-recognition)
- [Follow Controller](#follow-controller)
- [Safety-Oriented Control Logic](#safety-oriented-control-logic)
- [Camera System](#camera-system)
- [Model and Inference](#model-and-inference)
- [Configuration](#configuration)
- [Hardware Development](#hardware-development)
- [Flight Controller Integration](#flight-controller-integration)
- [Development Timeline](#development-timeline)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Vision Pipeline](#running-the-vision-pipeline)
- [Raspberry Pi Smoke Test](#raspberry-pi-smoke-test)
- [Testing](#testing)
- [Example Output](#example-output)
- [Project Images](#project-images)
- [Technical Design Notes](#technical-design-notes)
- [Limitations and Current Status](#limitations-and-current-status)
- [Future Development](#future-development)
- [License](#license)

---

## Overview

**ARGUS** is an autonomous-drone development project centered around onboard computer vision and target-following behavior.

The project was developed as a modular system so that the perception, tracking, gesture interpretation, and flight-control layers can be developed and tested independently.

The documented project progression includes:

1. Initial feasibility study for autonomous person/object tracking.
2. Raspberry Pi camera and computer-vision setup.
3. Real-time person detection and tracking.
4. Model refinement and Raspberry Pi implementation.
5. Custom 3D drone-component design and physical testing.
6. Evaluation of Raspberry Pi and Arduino UNO Q as companion-computing platforms.
7. UNO Q implementation and computer-vision integration.
8. Hardware assembly and power-system repair.
9. Pixhawk/PX4 calibration, ESC troubleshooting, and first-flight testing.

The repository contains both the **project-development documentation** and the **Python vision/control prototype**.

---

## Project Objective

The core technical objective is to build a drone system capable of:

- Detecting people in the camera view.
- Assigning session-local tracking IDs.
- Maintaining a selected target across frames.
- Interpreting simple hand gestures from pose keypoints.
- Converting target geometry into bounded movement intents.
- Providing a safe interface between computer vision and a future autopilot-control layer.
- Integrating the companion-computing system with a Pixhawk-based drone.
- Supporting future autonomous pursuit and gesture-triggered actions.

The system is deliberately modular: computer vision does not directly drive motors in the current Python implementation.

---

# System Architecture

```text
                         ┌───────────────────────┐
                         │   Camera / Video      │
                         │  Raspberry Pi Camera  │
                         └───────────┬───────────┘
                                     │ RGB frame
                                     ▼
                         ┌───────────────────────┐
                         │     Preprocessing     │
                         │  Resize + Letterbox   │
                         │  Normalize + CHW      │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │   YOLOv8n / ONNX      │
                         │    CPU Inference      │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │     Postprocessing    │
                         │ Person filtering      │
                         │ Box decoding + NMS    │
                         │ Optional keypoints    │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │    Person Tracker     │
                         │ IoU + centroid match  │
                         │ Session-local IDs     │
                         └───────────┬───────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
                    ▼                                 ▼
          ┌───────────────────┐             ┌───────────────────┐
          │ Gesture Recognizer│             │ Follow Controller │
          │ 17-point COCO     │             │ Target geometry   │
          │ pose keypoints    │             │ → bounded intent  │
          └─────────┬─────────┘             └─────────┬─────────┘
                    │                                 │
                    └────────────────┬────────────────┘
                                     ▼
                         ┌───────────────────────┐
                         │    Flight Intent      │
                         │ yaw / forward /       │
                         │ vertical / action     │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ Future Autopilot      │
                         │ / MAVLink Interface   │
                         │ (not motor control in │
                         │ current Python code)  │
                         └───────────────────────┘
```

---

# Current Software Implementation

The checked-in Python package is named `agni-vision` and is described as a Raspberry Pi person-tracking and gesture-intent pipeline.

The software is organized into independent modules:

| Module | Responsibility |
|---|---|
| `camera.py` | Camera abstraction for Picamera2/OpenCV |
| `preprocess.py` | YOLO letterbox preprocessing |
| `detector.py` | ONNX Runtime inference |
| `postprocess.py` | Detection decoding, person filtering, NMS, pose keypoints |
| `tracker.py` | Lightweight person tracking and session IDs |
| `gestures.py` | Debounced hand-gesture recognition |
| `controller.py` | Target-following and bounded flight-intent generation |
| `config.py` | Typed YAML configuration and validation |
| `main.py` | End-to-end command-line vision loop |

This separation allows individual components to be tested without requiring the complete drone hardware.

---

# Computer Vision Pipeline

The current pipeline follows this sequence:

```text
Camera Frame
    │
    ▼
RGB Image
    │
    ▼
Letterbox Resize
    │
    ▼
Normalized NCHW Tensor
    │
    ▼
YOLOv8n ONNX
    │
    ▼
Raw Detection Output
    │
    ▼
Person Class Filtering
    │
    ▼
Confidence Filtering
    │
    ▼
Non-Maximum Suppression
    │
    ▼
Person Detections
    │
    ▼
Lightweight Tracker
    │
    ▼
Track IDs
    │
    ├──────────────► Gesture Recognition
    │
    └──────────────► Follow Controller
                              │
                              ▼
                         Flight Intent
```

---

# Person Detection

The repository includes:

```text
code/models/yolov8n.onnx
code/models/yolov8n.pt
```

The ONNX model is loaded through **ONNX Runtime** using the CPU execution provider.

The detector:

1. Loads the ONNX model.
2. Reads its input shape dynamically.
3. Configures ONNX Runtime thread settings.
4. Enables graph optimization.
5. Receives a normalized image tensor.
6. Runs inference.
7. Returns the raw model output.

The post-processing layer then identifies the **person class** and converts the model's bounding-box representation into image coordinates.

---

# Preprocessing

The preprocessing module implements YOLO-style letterboxing.

For a camera frame:

```text
Original frame
     │
     ▼
Maintain aspect ratio
     │
     ▼
Resize to fit model input
     │
     ▼
Pad remaining area
     │
     ▼
Normalize 0–255 → 0–1
     │
     ▼
HWC → CHW
     │
     ▼
Add batch dimension
```

The current configuration uses a camera resolution of:

```text
640 × 480
```

and the bundled YOLOv8n model is expected to operate at:

```text
640 × 640
```

The preprocessing code also stores the scale and padding values so detections can later be mapped back to the original camera coordinates.

---

# Postprocessing

The post-processing stage supports two related output layouts:

### Detection output

YOLO detection output is interpreted as:

```text
x
y
width
height
class scores...
```

Only the **person class** is retained.

### Pose output

The code also supports a 56-value pose row:

```text
4 bounding-box values
+ 1 person confidence
+ 17 × (x, y, confidence)
```

This provides the keypoint data required by the gesture-recognition module.

### Non-Maximum Suppression

Overlapping person detections are filtered using IoU-based NMS.

The configured threshold is:

```yaml
iou_threshold: 0.45
```

---

# Person Tracking

The tracker is intentionally lightweight and does not depend on a large external tracking framework.

Each detected person receives a **session-local ID**:

```text
Person → Track ID
```

For example:

```text
Person A → ID 1
Person B → ID 2
Person C → ID 3
```

The tracker associates detections with existing tracks using:

- Bounding-box IoU.
- Normalized center-point distance.
- Detection-to-track matching.
- Missed-frame tolerance.

Important parameters:

```yaml
tracking:
  max_missed: 12
  min_hits: 2
  iou_threshold: 0.20
  max_center_distance: 0.18
```

A track becomes confirmed after the configured minimum number of successful hits.

A track can temporarily disappear and remain alive for a limited number of missed frames.

This is particularly useful when the target is briefly occluded.

---

# Gesture Recognition

The gesture module uses **17-point COCO pose keypoints**.

The implementation primarily uses:

```text
Nose
Left Shoulder
Right Shoulder
Left Wrist
Right Wrist
```

The basic gesture logic compares wrist positions with shoulder positions.

Supported gestures:

| Gesture | Intent |
|---|---|
| Right hand up | `start_follow` |
| Left hand up | `stop_follow` |
| Both hands up | `land` |

The recognizer includes:

- Keypoint confidence threshold.
- Hold-frame requirement.
- Per-track gesture state.
- Gesture cooldown.
- Track-specific gesture events.

This prevents a gesture from triggering immediately because of a single noisy frame.

Current configuration:

```yaml
gestures:
  hold_frames: 4
  cooldown_seconds: 2.0
  keypoint_threshold: 0.35
```

---

# Follow Controller

The follow controller converts target geometry into bounded movement **intents**.

It does not directly control motors.

The generated intent contains:

```json
{
  "yaw": 0.0,
  "forward": 0.0,
  "vertical": 0.0,
  "action": "follow",
  "target_id": 1,
  "reason": "target locked"
}
```

The controller considers:

### Horizontal error

```text
Target center ───────────────► Frame center
             horizontal error
```

Horizontal error is converted into a bounded yaw command.

### Vertical error

The target's vertical position is compared with the frame center and converted into a bounded vertical intent.

### Apparent target size

The target bounding-box area is compared with a desired area ratio.

This provides a simple distance proxy:

```text
Target too small  → move forward
Target near target size → hold
Target too large → reduce forward motion
```

Current controller limits:

```yaml
controller:
  target_area_ratio: 0.18
  deadband_x: 0.08
  deadband_y: 0.10
  max_yaw: 0.35
  max_forward: 0.30
  max_vertical: 0.25
  target_timeout_seconds: 0.75
```

---

# Safety-Oriented Control Logic

The controller intentionally uses conservative behavior.

### No target selected

```text
follow = false
        ↓
hover intent
```

### Target temporarily missing

The system does not immediately change the selected target.

### Target lost

If the target remains unavailable beyond the timeout:

```text
Target lost
    ↓
Safe hover intent
```

### Gesture target locking

Once a target is selected, gestures from another tracked person cannot silently take over the control session.

### Landing

The `both_hands_up` gesture can produce:

```text
action = "land"
```

when it belongs to the currently permitted target.

### Important

The current software outputs **high-level flight intent JSON**. It does not directly send PWM, motor commands, or MAVLink flight commands.

That separation is intentional and leaves the autopilot interface as a separate integration layer.

---

# Camera System

The camera abstraction supports two backends.

## Picamera2

Default Raspberry Pi backend:

```yaml
camera:
  backend: picamera2
```

This is intended for Raspberry Pi camera hardware.

## OpenCV

An OpenCV camera or video file can also be used:

```yaml
camera:
  backend: opencv
```

The implementation converts OpenCV's BGR frames to RGB before passing them into the model pipeline.

This makes local development and video-file testing possible without requiring the Raspberry Pi camera stack.

---

# Model and Inference

## YOLOv8n

The repository includes:

```text
models/yolov8n.onnx
models/yolov8n.pt
```

The ONNX version is used by the current runtime pipeline.

The detector uses:

```text
ONNX Runtime
CPUExecutionProvider
```

with configurable intra-operation threading.

Current model settings:

```yaml
model:
  path: models/yolov8n.onnx
  confidence: 0.45
  iou_threshold: 0.45
  threads: 4
  inference_stride: 2
```

### Inference stride

With:

```yaml
inference_stride: 2
```

the application performs model inference on every second frame.

This reduces computational load and can be useful on resource-constrained edge hardware.

---

# Configuration

All major runtime parameters are centralized in:

```text
code/configs/config.yaml
```

Configuration categories:

```text
model
├── model path
├── confidence threshold
├── IoU threshold
├── CPU threads
└── inference stride

camera
├── backend
├── source
├── width
└── height

tracking
├── missed-frame tolerance
├── confirmation hits
├── IoU threshold
└── center-distance threshold

gestures
├── hold frames
├── cooldown
├── keypoint confidence
└── gesture → action mapping

controller
├── target area
├── horizontal deadband
├── vertical deadband
├── yaw limit
├── forward limit
├── vertical limit
└── target timeout
```

The Python configuration layer uses dataclasses and validates important safety-related values before the application starts.

---

# Hardware Development

The physical drone development progressed alongside the software system.

## Main hardware elements documented during development

- Raspberry Pi computer-vision platform.
- Raspberry Pi camera.
- Arduino UNO Q experimentation and integration.
- Pixhawk flight controller.
- ESCs and motors.
- Power Distribution Board (PDB).
- Drone frame and 3D-printed components.
- 3D-printed structural prototypes.
- Suitable drone power and mounting hardware.

---

# 3D Design and Manufacturing

The project included custom 3D design and physical prototyping.

The development process included:

```text
Drone structural requirements
          ↓
3D component design
          ↓
Prototype printing
          ↓
Strength evaluation
          ↓
Design refinement
          ↓
Frame/component integration
```

A drone leg was initially printed as a prototype for strength and structural testing.

The project also encountered material constraints:

- PLA became temporarily unavailable during one stage.
- An alternative filament caused printing problems.
- Carbon-fibre-infused PETG was later considered for higher-strength components.
- Suitable screws and fasteners also affected the planned frame assembly.

To keep development moving, an already-available strong frame was used.

---

# Flight Controller Integration

The documented flight-control development used a **Pixhawk**.

The project experimented with:

- QGroundControl.
- PX4 firmware.
- ArduPilot firmware.
- Mission Planner.

## PX4 / QGroundControl

The successful configuration path used:

```text
Pixhawk
   ↓
QGroundControl
   ↓
PX4 firmware
   ↓
Calibration
   ↓
ESC/motor connection verification
   ↓
Flight preparation
```

## ESC troubleshooting

Incorrect ESC connections were identified during setup and corrected before flight testing.

## ArduPilot / Mission Planner

ArduPilot firmware and Mission Planner were also tested, but the documented calibration attempt did not succeed.

The project then returned to:

```text
PX4 + QGroundControl
```

for further configuration and calibration.

---

# Development Timeline

## Week 0 — Ideation and Feasibility

The project concept was established around:

- Person/object detection.
- Target tracking.
- Autonomous pursuit.
- Gesture-based actions.
- Pixhawk-based flight control.

The initial feasibility study compared Raspberry Pi and Arduino UNO Q in terms of compute capability, power, camera support, real-time behavior, and software ecosystem.

The project also established an initial safety constraint: autonomous flight experiments should begin in a tethered or controlled environment.

---

## Week 1 — Computer Vision Setup

The Raspberry Pi vision environment was established.

Work included:

- Camera configuration.
- Live video acquisition.
- Computer-vision dependencies.
- Initial person detection.
- Tracking IDs.
- FPS/frame monitoring.
- Groundwork for hand-gesture recognition.

Initial camera testing demonstrated multiple detected people with bounding boxes and tracking IDs.

![Initial Raspberry Pi computer vision test](docs/pics1/WhatsApp%20Image%202026-09-17%20at%2011.48.52%20PM.jpeg)

---

## Week 2 — Model Refinement and 3D Design

The computer-vision model was deployed and refined on the Raspberry Pi.

Parallel mechanical development began with:

- Drone component design.
- Dimension planning.
- Mounting considerations.
- Assembly compatibility.
- Preparation for 3D printing.

---

## Week 3 — Raspberry Pi Implementation and Prototype Testing

The Raspberry Pi implementation was completed for the computer-vision system.

Mechanical development progressed toward physical testing.

A 3D-printed drone leg was produced as an initial structural prototype for strength evaluation.

---

## Week 4 — 3D Refinement and Drone Physics

The overall 3D-printed drone design was refined.

The development also considered:

- Thrust.
- Lift.
- Weight.
- Forces acting on the drone.
- Structural requirements.

A filament-availability problem caused printing delays and required printer cleaning before manufacturing could continue.

![3D printed drone design](docs/pics1/WhatsApp%20Image%202026-09-17%20at%2011.51.14%20PM.jpeg)

---

## Week 5 — Project Continuity and Development Handoff

The project documentation records a transition in project participation and a technical briefing covering:

- Previous software work.
- 3D design.
- Printing.
- Drone physics.
- Current project state.
- Remaining development.

No teammate-specific information is included in this README.

---

## Week 6 — Frame Selection and UNO Q

Mechanical work focused on frame material and hardware availability.

The planned carbon-fibre-infused PETG approach was affected by material availability, while PLA did not provide the required guaranteed strength for the intended application.

A strong available frame was therefore selected.

In parallel, work began on the Arduino UNO Q implementation.

---

## Week 7 — UNO Q Debugging and Computer Vision

The Arduino UNO Q initially required troubleshooting and reset procedures.

The Raspberry Pi vision code was adapted toward UNO Q compatibility with AI-assisted code conversion and subsequent corrections.

The documented development then reached a stage where computer vision was implemented on the UNO Q.

![UNO Q computer vision implementation](docs/pics1/WhatsApp%20Image%202026-09-17%20at%2011.52.14%20PM.jpeg)

---

## Week 8 — Hardware Assembly and PDB Replacement

The UNO Q was integrated into the drone hardware.

Major electronic components were positioned and soldered.

During power-up testing, the PDB failed.

The documented cause was the absence of a required capacitor. The PDB was replaced, a suitable capacitor was added, and the power-system connections were completed.

![Drone hardware assembly](docs/pics1/WhatsApp%20Image%202026-09-17%20at%2011.53.48%20PM.jpeg)

---

## Week 9 — Pixhawk Calibration and First Flight

The final documented development stage focused on:

- Pixhawk configuration.
- PX4 firmware flashing.
- QGroundControl calibration.
- ESC connection correction.
- ArduPilot/Mission Planner experimentation.
- PX4 reflash and recalibration.
- First-flight preparation.

After resolving the documented configuration and connection issues, the drone completed its first flight.

![First flight](docs/pics1/WhatsApp%20Image%202026-09-17%20at%2011.57.30%20PM.jpeg)

---

# Repository Structure

```text
InfiniteInferno/
│
├── README.md
├── LICENSE
│
├── code/
│   ├── configs/
│   │   └── config.yaml
│   │
│   ├── models/
│   │   ├── yolov8n.onnx
│   │   └── yolov8n.pt
│   │
│   ├── scripts/
│   │   └── pi_smoke_test.py
│   │
│   ├── src/
│   │   ├── __init__.py
│   │   ├── camera.py
│   │   ├── config.py
│   │   ├── controller.py
│   │   ├── detector.py
│   │   ├── gestures.py
│   │   ├── main.py
│   │   ├── postprocess.py
│   │   ├── preprocess.py
│   │   └── tracker.py
│   │
│   ├── tests/
│   │   ├── test_camera.py
│   │   ├── test_config.py
│   │   ├── test_detector.py
│   │   ├── test_model.py
│   │   └── test_tracker.py
│   │
│   ├── requirements.txt
│   └── pyproject.toml
│
└── docs/
    ├── week-00.md
    ├── Gesture_Drone_Week_1_ (2).md
    ├── Week_2_Model_Refinement_and_3D_Design.md
    ├── Week_3_Raspberry_Pi_Implementation_and_3D_Design_Testing.md
    ├── Week_4_Drone_Project_Log.md
    ├── Week_5_Drone_Project_Log.md
    ├── Week_6_Drone_Project_Log.md
    ├── Week_7_Drone_Project_Log.md
    ├── Week_8_Drone_Project_Log.md
    ├── Week_9_Drone_Project_Log.md
    │
    └── pics1/
        ├── WhatsApp Image 2026-09-17 at 11.48.52 PM.jpeg
        ├── WhatsApp Image 2026-09-17 at 11.51.14 PM.jpeg
        ├── WhatsApp Image 2026-09-17 at 11.52.14 PM.jpeg
        ├── WhatsApp Image 2026-09-17 at 11.53.48 PM.jpeg
        └── WhatsApp Image 2026-09-17 at 11.57.30 PM.jpeg
```

---

# Requirements

The current Python project requires:

- Python `>= 3.11`
- NumPy `>= 1.26, < 3`
- ONNX Runtime `>= 1.19, < 2`
- PyYAML `>= 6, < 7`

On Raspberry Pi OS, the project expects the platform-provided packages for:

- Picamera2.
- OpenCV.

The repository's `requirements.txt` intentionally does not install those two platform packages through pip.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/BuilderinResidencyTinkerspace/InfiniteInferno.git
cd InfiniteInferno
```

Enter the Python project:

```bash
cd code
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install the Python dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Raspberry Pi OS, ensure the system packages for Picamera2 and OpenCV are installed through the operating system package manager.

---

# Running the Vision Pipeline

From the `code/` directory:

```bash
python -m src.main
```

The default configuration uses:

```text
Camera backend: Picamera2
Resolution:      640 × 480
Model:           YOLOv8n ONNX
Inference stride: 2
```

---

## OpenCV Camera

For an OpenCV camera:

```bash
python -m src.main --camera opencv --source 0
```

A video file can also be supplied as the OpenCV source:

```bash
python -m src.main --camera opencv --source path/to/video.mp4
```

---

## Explicit Target Selection

A known track ID can be selected:

```bash
python -m src.main --target-id 1
```

The controller will only follow the selected session-local track.

---

## Automatic First Target

For development/testing:

```bash
python -m src.main --auto-follow
```

The largest first visible confirmed person is selected.

---

## Local Display

To show an annotated OpenCV preview:

```bash
python -m src.main --display
```

This requires a graphical OpenCV environment.

---

## Limit the Number of Frames

For controlled testing:

```bash
python -m src.main --max-frames 100
```

---

# Raspberry Pi Smoke Test

The repository includes:

```text
code/scripts/pi_smoke_test.py
```

Run:

```bash
cd code
python scripts/pi_smoke_test.py
```

This performs a hardware-free model inference test and reports:

- CPU architecture.
- Operating system.
- Python version.
- Model input size.
- Model output shape.
- Overall status.

Example format:

```json
{
  "machine": "aarch64",
  "os": "...",
  "python": "...",
  "model_input": [640, 640],
  "model_output": [1, 84, 8400],
  "status": "ok"
}
```

The exact output shape depends on the model file being used.

---

# Testing

The project includes unit tests covering:

### Camera

- Invalid backend handling.
- OpenCV stream failure behavior.

### Detector

- Model input-shape handling.
- Injected inference session testing.

### Preprocessing and postprocessing

- Letterbox geometry.
- Coordinate restoration.
- NMS.
- Pose-keypoint extraction.

### Tracker

- ID persistence during motion.
- Brief occlusion handling.
- Multiple-person ID assignment.

### Gestures

- Hold-frame debounce.
- Gesture cooldown.
- Both-hands priority.

### Controller

- Target selection.
- Target loss behavior.
- Landing gesture handling.
- Protection against a bystander taking control.

Run the complete test suite:

```bash
cd code
python -m pytest
```

---

# Example Output

The main application emits machine-readable JSON lines.

A representative intent looks like:

```json
{
  "frame": 120,
  "fps": 18.7,
  "track_ids": [1, 2],
  "gestures": [],
  "intent": {
    "yaw": 0.12,
    "forward": 0.08,
    "vertical": -0.03,
    "action": "follow",
    "target_id": 1,
    "reason": "target locked"
  }
}
```

When following is disabled, the controller returns a safe default intent:

```json
{
  "yaw": 0.0,
  "forward": 0.0,
  "vertical": 0.0,
  "action": "hover",
  "target_id": null,
  "reason": "follow mode disabled"
}
```

---

# Project Images

## Raspberry Pi Computer Vision

![Raspberry Pi computer vision](docs/pics1/WhatsApp%20Image%202026-09-17%20at%2011.48.52%20PM.jpeg)

Initial person detection and tracking using the Raspberry Pi camera.

## 3D-Printed Drone Design

![3D printed drone design](docs/pics1/WhatsApp%20Image%202026-09-17%20at%2011.51.14%20PM.jpeg)

Physical development of the drone structure.

## UNO Q Computer Vision

![UNO Q computer vision](docs/pics1/WhatsApp%20Image%202026-09-17%20at%2011.52.14%20PM.jpeg)

Documented computer-vision implementation on the Arduino UNO Q.

## Hardware Assembly

![Hardware assembly](docs/pics1/WhatsApp%20Image%202026-09-17%20at%2011.53.48%20PM.jpeg)

Drone hardware assembly and component integration.

## First Flight

![First flight](docs/pics1/WhatsApp%20Image%202026-09-17%20at%2011.57.30%20PM.jpeg)

Documented first-flight milestone.

---

# Technical Design Notes

## Edge Computing

The vision stack is designed for onboard/edge processing rather than requiring a remote server.

This reduces the dependency on network connectivity for the perception pipeline and allows the camera, detector, tracker, and control-intent generation to operate locally.

## Lightweight Model

YOLOv8n was selected as a lightweight detection model suitable for experimentation on resource-constrained hardware.

The use of ONNX Runtime provides a deployment-oriented inference path.

## Modular Tracking

Tracking is implemented separately from detection.

This means:

```text
Detector
   ↓
Detections
   ↓
Tracker
   ↓
Persistent IDs
```

A detector update does not have to equal a new target identity.

## Gesture Debouncing

Gestures require multiple consecutive frames before an event is emitted.

This reduces false triggers caused by:

- Keypoint noise.
- Temporary pose changes.
- Single-frame misdetections.

## Bounded Control

The follow controller clamps movement values to configured limits.

This provides a defined interface for a future autopilot layer rather than allowing raw image geometry to directly generate unrestricted commands.

---

# Limitations and Current Status

The repository should be understood as a **development-stage autonomous-drone system**, not as a complete production flight-control stack.

### Current Python implementation

The checked-in vision software currently provides:

- Raspberry Pi camera support.
- OpenCV camera/video support.
- YOLOv8n ONNX inference.
- Person detection.
- Detection filtering and NMS.
- Lightweight person tracking.
- Track IDs.
- Pose-keypoint handling.
- Hand-gesture recognition.
- Target selection.
- Bounded follow intents.
- JSON output.
- Unit tests.
- Raspberry Pi model smoke testing.

### Not currently implemented in this Python stack

The current repository code does **not** directly implement:

- Motor PWM control.
- ESC control.
- Direct MAVLink commands.
- Direct Pixhawk command transmission.
- Autonomous flight stabilization.
- Full obstacle avoidance.
- Production-grade visual-inertial navigation.
- GPS-based autonomous navigation.

Those functions belong to the flight-controller/autopilot integration layer and are separate from the current vision-intent prototype.

### Project documentation vs. current code

The weekly documentation records the broader project evolution, including UNO Q implementation, physical drone assembly, Pixhawk calibration, and first flight.

The `code/` directory is specifically the software vision/control prototype and should not be interpreted as containing the complete hardware or autopilot implementation.

---

# Future Development

The modular architecture provides a path toward:

```text
Current
  │
  ├── Person detection
  ├── Tracking
  ├── Gesture recognition
  └── Flight intent generation
          │
          ▼
Future integration
          │
          ├── MAVLink interface
          ├── Pixhawk command layer
          ├── Flight-state feedback
          ├── Target reacquisition
          ├── Obstacle avoidance
          ├── Better pose/gesture estimation
          ├── Hardware-accelerated inference
          └── Full autonomous mission logic
```

Potential optimization areas include:

- Model quantization.
- Hardware-accelerated inference.
- More efficient pose estimation.
- Better multi-object association.
- Improved target-reacquisition logic.
- Flight-controller feedback integration.
- Sensor fusion.
- Robust failsafe handling.
- Controlled outdoor flight testing.

---

# Development Philosophy

ARGUS is structured around a separation of concerns:

```text
PERCEPTION
Camera → Detection → Tracking

INTERPRETATION
Tracking + Pose → Gesture Events

DECISION
Target Geometry + Gesture Events → Flight Intent

AUTOPILOT
Flight Intent → Future MAVLink/Pixhawk Interface

ACTUATION
Autopilot → ESCs → Motors
```

Keeping these layers separate makes it easier to test the computer-vision system independently from the flight hardware and reduces the risk of coupling experimental vision code directly to motor control.

---

# Documentation

The complete technical development history is available in the [`docs/`](docs/) directory.

Weekly records:

- [`Week 0 — Ideate`](docs/week-00.md)
- [`Week 1 — Computer Vision Setup`](docs/Gesture_Drone_Week_1_%20(2).md)
- [`Week 2 — Model Refinement and 3D Design`](docs/Week_2_Model_Refinement_and_3D_Design.md)
- [`Week 3 — Raspberry Pi Implementation and 3D Design Testing`](docs/Week_3_Raspberry_Pi_Implementation_and_3D_Design_Testing.md)
- [`Week 4 — 3D Print Refinement and Drone Physics`](docs/Week_4_Drone_Project_Log.md)
- [`Week 5 — Project Briefing`](docs/Week_5_Drone_Project_Log.md)
- [`Week 6 — Frame Selection and UNO Q`](docs/Week_6_Drone_Project_Log.md)
- [`Week 7 — UNO Q Debugging and Computer Vision`](docs/Week_7_Drone_Project_Log.md)
- [`Week 8 — Hardware Assembly and PDB Replacement`](docs/Week_8_Drone_Project_Log.md)
- [`Week 9 — Pixhawk Calibration and First Flight`](docs/Week_9_Drone_Project_Log.md)

---

# License

This project is distributed under the license included in the repository.

See [`LICENSE`](LICENSE) for the complete license text.
