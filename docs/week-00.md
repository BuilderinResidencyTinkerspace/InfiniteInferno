# Week 0 — Ideate

**Goal this week:**
Land on a project idea and validate whether our planned hardware stack can realistically support it.

---

## What we did

* Brainstormed multiple hardware project directions for the 9-week build.
* Finalized the concept:
  **FALCON EYE** — an autonomous drone that:

  * Visually tracks a person/object (bounding-box based detection)
  * Pursues the target
  * Maps hand gestures to onboard actions (e.g., fire suppression or target-specific actions)
* Defined scope intentionally loosely:

  * Tracking, pursuit, and gesture-mapped actions treated as modular features
  * Allows partial implementation instead of a rigid full-stack dependency
* Evaluated feasibility of:

  * **Pixhawk + Arduino UNO Q + camera stack**
  * Considered compute requirements, power draw, and payload constraints
* Compared **Arduino UNO Q vs Raspberry Pi** across:

  * Compute capability
  * Power consumption
  * Camera/ISP support
  * Real-time control ability
  * Software ecosystem maturity

---

## Problems and blockers

* Neither Arduino UNO Q nor Raspberry Pi (standalone) can run heavy real-time detection models (e.g., full YOLO) at high FPS.
* Viable approach requires:

  * Lightweight models (YOLOv8n / MobileNet-SSD)
  * Reduced resolution for real-time performance on a moving drone
* Arduino UNO Q:

  * Very new board with limited documentation
  * Sparse examples for MAVLink integration and computer vision workloads
* Raspberry Pi:

  * Lacks a real-time MCU
  * Time-critical actions (gesture triggers, actuation) require an additional microcontroller

---

## Decisions

* Finalized project scope:

  * **Core:** Person/object tracking + autonomous pursuit via Pixhawk
  * **Stretch goals:** Gesture-mapped actions and fire/target-response mechanisms
* Deferred final choice of companion computer (UNO Q vs Raspberry Pi) to Week 1 after hands-on testing
* Established safety constraint:

  * All autonomous flight testing must be **tethered or caged initially**

---

## Next week

* Prototype full pipeline:

  * Pixhawk + companion computer + camera
* Validate:

  * Real-time detection performance
  * Latency and control loop feasibility
  * Integration gaps between vision and flight control

---

## Links

* **Code:**
* **Photos / CAD:**
