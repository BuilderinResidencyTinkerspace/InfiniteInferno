# Week 1 — Raspberry Pi Prototyping

**Goal this week:**
Start implementing the companion computer + Pixhawk stack and validate it in practice.

---

## What we did

* Began implementation using **Raspberry Pi + Pixhawk** as the companion computer setup:

  * Chosen for its mature ecosystem and well-documented MAVLink companion computer workflow
  * Strong support for OpenCV and vision-based pipelines
* Set up the initial **MAVLink communication path** between Pixhawk and Raspberry Pi
* Started scoping the **vision pipeline** on the Pi for object/person detection
* Evaluated requirements for integrating:

  * Gesture-mapped actions
  * Time-critical triggers (e.g., fire-suppression actuation)

---

## Problems and blockers

* **No onboard real-time MCU on Raspberry Pi:**

  * Identified need for a separate microcontroller to handle latency-sensitive tasks
  * Required for reliable gesture-triggered or time-critical actions
* This introduces additional complexity:

  * Extra hardware (MCU board)
  * Additional communication layer (UART/USB)
  * Increased system latency between vision → decision → actuation
  * More potential failure points in the pipeline
* Overall architecture started to feel **fragmented and less robust**

---

## Decisions

* Move away from the **Raspberry Pi + separate MCU** architecture
* Explore alternative that can handle:

  * Vision processing (Linux side)
  * Real-time control (MCU side)
  * Within a **single integrated board**
* Identified **Arduino UNO Q** as a promising candidate:

  * Combines **QRB2210 MPU + STM32U585 MCU**
  * Built-in **RPC bridge** between Linux and MCU domains
  * Potential to reduce latency and simplify system design

---

## Next week

* Build and validate stack using **Arduino UNO Q**
* Test:

  * MAVLink communication with Pixhawk
  * Camera interfacing and capture pipeline
  * Basic real-time vision inference feasibility
* Evaluate whether UNO Q can replace Pi fully in the system

---

## Links

* **Code:**
* **Photos / CAD:**


