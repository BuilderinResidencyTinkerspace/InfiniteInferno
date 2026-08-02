Week 0 — Ideate

Goal this week: Land on a project idea and validate whether our planned hardware stack can realistically support it.

What we did

Brainstormed hardware project directions for the 9-week build.
Landed on: an autonomous drone that visually tracks a person/object (bounding-box based detection), pursues it, and can map hand gestures to onboard actions (e.g. a fire-suppression/extinguishing action, or other target-spot actions).
Named the project FALCON EYE (team: Infinite Inferno).
Scoped the idea deliberately loose — treating tracking, pursuit, and gesture-mapped actions as independent features we may implement partially, not a fixed spec.
Investigated whether a Pixhawk + Arduino UNO Q + camera stack could realistically handle this, factoring in power draw, onboard compute for running a vision model, and physical size/weight constraints on a small drone frame.
Ran a comparison of Arduino UNO Q vs Raspberry Pi as the companion computer, across compute power, power draw, camera/ISP support, real-time control capability, and software ecosystem maturity.

Problems and blockers

Neither board alone (without an accelerator) can run a heavy real-time detector (e.g. full YOLO) at high fps — both would need lightweight models (YOLOv8n/MobileNet-SSD) at reduced resolution to be usable on a moving platform.
UNO Q is a newly launched board with limited documentation/community examples for MAVLink and CV workloads specifically.
Raspberry Pi has no onboard real-time MCU, so gesture-triggered or time-critical actions would need a separate microcontroller.

Decisions

Proceed with idea scoping as: core = person/object tracking + pursuit via Pixhawk; stretch goals = fire/target-spot action and gesture-mapped actions.
Defer final companion-computer choice (UNO Q vs Pi) to Week 1, after hands-on testing.
Any autonomous flight testing must be tethered/caged first — safety-first build order.

Next week

Prototype the Pixhawk + companion computer + camera pipeline hands-on and see where the real gaps are.
