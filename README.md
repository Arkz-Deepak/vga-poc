# VGA PoC: Vision-Geometry-Action Model (Robotics VLA)

Lightweight ($\le 0.5\text{B}$ parameter) Proof of Concept (PoC) of the Vision-Geometry-Action (VGA) model evaluated on 3 spatial manipulation tasks from the **LIBERO** benchmark using Hugging Face's **LeRobot** framework.

---

## 🎯 What is this Project? (Demystifying the Terminology)

- **Not Biology / Not DNA**: This is a **VLA (Vision-Language-Action)** robotics AI system. (In audio/calls, *V-L-A* is often misheard or transcribed as *DNA*!).
- **Vision (V)**: The robot looks at the scene through an RGB camera (e.g. tabletop with bowls, plates, soup cans).
- **Language (L)**: The robot receives natural language instructions (e.g. *"pick up the black bowl and place it on the plate"*).
- **Action (A)**: The robot outputs smooth 3D trajectories to control its arm and gripper ($[\Delta x, \Delta y, \Delta z, r_x, r_y, r_z, \text{gripper}]$).
- **Geometry (G)**: Unlike standard 2D flat VLAs, VGA adds **Ray-RoPE** (physical 3D camera ray embedding) to give the model true 3D spatial grounding with **zero runtime overhead**.

---

## 🏗️ Architecture Overview

1. **Vision Backbone & Projector**:
   - SigLIP-B/16 (LoRA tuned on blocks 6–12)
   - `UnifiedSpaceToDepthProjector`: Hardware-accelerated pixel unshuffle (3×) compressing 576 patch tokens into 64 visual tokens.
2. **Language Backbone**:
   - Pruned 12-layer SmolLM2 model (~181M params) for fast language conditioning.
3. **Zero-Overhead 3D Grounding**:
   - `CentroidRayRoPE`: Projects camera origin $t_{base}$ and 3D unit viewing rays for the $8 \times 8$ grid into language space with sub-millisecond pre-pass attention.
4. **Action Expert & Controller**:
   - 12-layer Diffusion Transformer (DiT) / Flow Matching model predicting 16 future waypoints at 50 Hz using a 4-step Euler solver.
   - `SchmittTriggerGripper`: Affine mapping and hysteresis deadband ($0.35 / 0.65$) to eliminate gripper chatter.
5. **Kinematic & Physics Regularization**:
   - Safe Lie algebra axis-angle extraction with 4th-order Taylor series guard (zero `NaN` near $\theta = 0$).
   - Finite-difference acceleration ($L_{acc}$) and jerk ($L_{jerk}$) penalties for smooth human-like motion.

---

## 🚀 Quickstart: Phase 1 Local Verification

```bash
# 1. Activate the environment
source .venv/bin/activate

# 2. Run the synthetic tensor sanity checks
python scripts/verify_phase1.py
```
