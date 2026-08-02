# Week 2 — Arduino UNO Q Stack

**Goal this week:**
Build the initial hardware/software stack around Arduino UNO Q + Pixhawk + camera.

---

## What we did

* Transitioned the companion computer to **Arduino UNO Q** and began building the full system stack:

  * **Pixhawk** for flight control
  * **UNO Q (MPU/Linux side)** for vision processing
  * **UNO Q (STM32 MCU side)** for real-time gesture/action handling
  * **Camera** for visual input
* Initiated **camera integration** on the UNO Q Linux environment as the first step toward the detection pipeline
* Began testing the **RPC bridge** between MPU and MCU:

  * Validated feasibility of passing data/commands from vision → real-time action layer
  * Confirmed potential for single-board coordination of perception and actuation

---

## Problems and blockers

* **Limited ecosystem maturity:**

  * UNO Q’s AI/vision tooling is still emerging
  * Less documentation and fewer community resources compared to Raspberry Pi
  * Required trial-and-error for stable camera + inference pipeline setup
* **Unvalidated real-world constraints:**

  * Power consumption under sustained vision workload not yet characterized
  * Thermal performance on the drone airframe still unknown

---

## Decisions

* Committed to **Arduino UNO Q (4GB variant)** as the primary companion computer

  * Utilize onboard MCU for real-time gesture/action handling
  * Avoid adding a separate microcontroller
* Adopt modular validation approach:

  * First ensure **MAVLink command path works independently**
  * Then validate **basic detection pipeline**
  * Fuse both systems only after individual stability is confirmed

---

## Next week

* Continue building **detection + MAVLink integration pipeline**
* Begin **bench testing**:

  * Vision output → flight command mapping
  * Latency and responsiveness of control loop
* Identify bottlenecks in real-time performance before flight testing

---

## Links

* **Code:**
* **Photos / CAD:**
