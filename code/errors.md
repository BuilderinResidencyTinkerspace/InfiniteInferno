# Error Log & Troubleshooting Guide: InfiniteInferno on Raspberry Pi

This document serves as a comprehensive reference guide for resolving common execution, dependency, and hardware interface errors encountered while running the `InfiniteInferno` repository on a modern Raspberry Pi (Bookworm/Bullseye OS) with the official CSI camera module.

---

## Error 1: Relative Import Issues in Execution Script

**The Error:**
```text
Traceback (most recent call last):
  File "/home/raspi/InfiniteInferno/code/src/main.py", line 11, in <module>
    from .camera import Camera
ImportError: attempted relative import with no known parent package
```

**The Cause:**
The script `src/main.py` utilizes relative imports (`from .camera import Camera`), but is being executed directly via `python main.py`. Python forbids relative imports in scripts executed directly without a parent package context (`-m`).

**The Fix:**
1. Stripped the leading dots (`.`) from top-level imports in `main.py` using `sed` or a text editor (e.g., changing `from .camera import Camera` to `from camera import Camera`).
2. Executed the script by dynamically adding `src` to the `PYTHONPATH` rather than moving into the directory:
```bash
cd /home/raspi/InfiniteInferno/code
PYTHONPATH=src python src/main.py
```

---

## Error 2: Missing Camera Backend Dependencies (OpenCV & Picamera2)

**The Error (Picamera2):**
```text
RuntimeError: Picamera2 is unavailable; install python3-picamera2 or use --camera opencv
```
**The Error (OpenCV):**
```text
RuntimeError: OpenCV is unavailable; install python3-opencv
```

**The Cause:**
The active Python Virtual Environment (`venv`) was completely isolated from the host OS system packages. Both `picamera2` and `cv2` (OpenCV) require low-level system C++ bindings (libcamera) to function properly on a Raspberry Pi.

**The Initial Fix (Failed OpenCV Headless approach):**
Installing `opencv-python-headless` resolved the module error but resulted in silent crashes because the headless version entirely strips out GUI/Display rendering capabilities.

**The Proper Fix (System Packages Setup):**
Destroy the isolated `venv`, install the native Raspberry Pi OS optimized camera libraries via `apt`, and rebuild the virtual environment to inherit system site packages.

```bash
# 1. Install optimized system libraries
sudo apt update
sudo apt install -y python3-picamera2 python3-libcamera python3-opencv

# 2. Rebuild virtual environment with system package access
rm -rf venv
python3 -m venv venv --system-site-packages
source venv/bin/activate
```

---

## Error 3: OpenCV Silent Exit (The `/dev/video*` Issue)

**The Issue:**
Running `main.py` with `--camera opencv` resulted in the ONNX runtime initializing, but the script immediately exiting without an error trace. 

**The Cause:**
Modern Raspberry Pi OS uses the `libcamera` stack, which generates dozens of virtual device endpoints (`/dev/video0` through `/dev/video35`). OpenCV blind-grabs `/dev/video0`, fails to decode the raw libcamera stream, assumes the camera is inactive, and gracefully exits the main loop.

**The Fix:**
For an official Raspberry Pi CSI ribbon camera on modern Pi OS, the OpenCV backend must be avoided. The `--camera picamera2` argument must be utilized instead, requiring the system package fixes implemented in Error 2.

---

## Error 4: X11 Display Routing 

**The Error:**
The `--display` flag was passed, but the script exited or failed to render a graphical window on the connected monitor.

**The Cause:**
The terminal instance running the Python script lacked the environmental context to target the primary X11/Wayland display output.

**The Fix:**
Manually define the `$DISPLAY` environment variable prior to execution:
```bash
export DISPLAY=:0
```

---

## Final Working Execution Command Sequence

After resolving all import paths, system dependencies, display routing, and `libcamera` interfacing, this is the final validated sequence to successfully launch the prototype:

```bash
cd /home/raspi/InfiniteInferno/code
source venv/bin/activate

# Install any remaining pip requirements (like onnxruntime)
pip install -r requirements.txt

# Bind display and execute with picamera2 backend
export DISPLAY=:0
PYTHONPATH=src python src/main.py --camera picamera2 --display --auto-follow
```
