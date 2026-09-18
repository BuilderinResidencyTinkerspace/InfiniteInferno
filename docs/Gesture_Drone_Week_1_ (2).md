# Week 1 --- Computer Vision Setup

## Goal this week:

Set up the **computer vision system on the Raspberry Pi** and establish
a working camera and real-time video-processing pipeline as the
foundation for gesture-based drone control.

------------------------------------------------------------------------

## What we did

-   Set up the **Raspberry Pi environment** required for computer vision
    development.
-   Connected and configured the **Raspberry Pi camera** for live video
    capture.
-   Installed and configured the required software, libraries, and
    dependencies.
-   Tested the camera feed and verified **real-time video acquisition**.
-   Implemented initial **person detection and tracking** using the
    camera feed.
-   Verified detection using **bounding boxes and tracking IDs**.
-   Monitored real-time parameters such as **FPS, frame number, tracking
    IDs, and detected actions**.
-   Worked extensively on troubleshooting the computer vision setup and
    resolving configuration and compatibility issues.
-   Studied the requirements for extending the detection pipeline toward
    **hand-gesture recognition** for drone control.

### Initial Computer Vision Testing

The Raspberry Pi camera was successfully tested with the computer vision
pipeline. The system was able to process the live camera feed and detect
multiple people, displaying bounding boxes and individual tracking IDs.
This provided an initial working foundation for developing the
gesture-recognition system.

![Raspberry Pi Computer Vision Testing](week_1.jpeg)

*Figure 1: Initial real-time person detection and tracking test using
the Raspberry Pi camera.*

------------------------------------------------------------------------

## Problems and Blockers

-   Setting up the computer vision environment required troubleshooting
    software dependencies and configurations.
-   Camera configuration and reliable video acquisition required
    repeated testing.
-   Real-time processing performance on the Raspberry Pi needs further
    optimization.
-   The existing detection and tracking system needs to be extended to
    recognize **specific hand gestures**.
-   Reliable gesture recognition is required before integrating the
    system with drone control commands.

------------------------------------------------------------------------

## Decisions

-   Use the **Raspberry Pi as the primary computer vision processing
    platform**.
-   Continue developing the camera-based vision pipeline before
    integrating it with the drone control system.
-   Build the gesture-recognition system on top of the tested real-time
    detection and tracking pipeline.
-   Focus on achieving reliable gesture detection before mapping
    gestures to drone movements.

------------------------------------------------------------------------

## Links

-   **Code:**\
-   **Photos / Documentation:**\
-   **CAD / Project Files:**

------------------------------------------------------------------------

## Weekly Log

**Week 1:** Raspberry Pi computer vision setup, camera configuration,
initial real-time person detection and tracking, troubleshooting, and
groundwork for implementing hand-gesture recognition.
