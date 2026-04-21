# Tracker Guide

motcpp ships 10 state-of-the-art multi-object tracking algorithms behind a single unified API. This page documents every tracker: what it does, when to use it, all constructor parameters, and a working code snippet.

## Overview

| Tracker | Type | HOTA ↑ | MOTA ↑ | IDF1 ↑ | FPS ↑ |
|---------|------|:------:|:------:|:------:|:-----:|
| [SORT](#sort) | Motion | 62.4 | 75.2 | 69.2 | **1250** |
| [ByteTrack](#bytetrack) | Motion | 66.5 | 76.4 | 77.6 | 1100 |
| [OC-SORT](#oc-sort) | Motion | 64.6 | 73.9 | 74.4 | 850 |
| [UCMCTrack](#ucmctrack) | Motion | 64.0 | 75.6 | 73.9 | 980 |
| [OracleTrack](#oracletrack) | Motion | 66.9 | 77.3 | 79.7 | 449 |
| [DeepOC-SORT](#deepoc-sort) | ReID | 65.8 | 75.1 | 76.2 | 120 |
| [StrongSORT](#strongsort) | ReID | 66.2 | 75.8 | 77.1 | 95 |
| [BoT-SORT](#bot-sort) | ReID | 66.8 | 76.2 | 78.3 | 85 |
| [HybridSORT](#hybridsort) | ReID | 66.4 | 76.0 | 77.8 | 90 |
| [BoostTrack](#boosttrack) | ReID | **67.5** | **77.1** | **79.2** | 75 |

> Benchmarked on MOT17-ablation with YOLOX detections + FastReID embeddings. FPS measured on Intel i9-13900K, single thread.

## Unified API

Every tracker exposes the same interface:

```cpp
// Input:  dets  — N×6  [x1, y1, x2, y2, confidence, class_id]
//         img   — OpenCV frame (used by CMC and ReID)
//         embs  — N×D  pre-computed embeddings (optional)
// Output: tracks — M×8 [x1, y1, x2, y2, track_id, conf, class_id, det_idx]
Eigen::MatrixXf tracks = tracker.update(dets, img);

// Reset all state between sequences
tracker.reset();
```

---

## Motion-Only Trackers

Motion-only trackers use Kalman filtering and IoU-based association. No appearance model is needed — fast, simple, and highly effective.

---

### SORT

**Simple Online and Realtime Tracking** — the foundational tracker from which nearly every modern algorithm descends.

**How it works:**
SORT maintains a Kalman filter per track with state `[x, y, s, r, ẋ, ẏ, ṡ]` (center, scale, aspect ratio, velocities). Each frame it predicts all tracks forward, computes the IoU matrix against new detections, and solves the linear assignment problem (Hungarian algorithm). Unmatched tracks age out after `max_age` frames; new detections become tentative tracks confirmed after `min_hits`.

**When to use:**
- Maximum throughput (>1000 FPS) is required
- Simple scenes with low occlusion density
- Baseline comparisons

```cpp
#include <motcpp/trackers/sort.hpp>

motcpp::trackers::Sort tracker(
    0.3f,   // det_thresh   — minimum detection confidence to consider
    1,      // max_age      — frames without a match before track deletion
    50,     // max_obs      — observation history length per track
    3,      // min_hits     — detections needed before a track is confirmed
    0.3f,   // iou_threshold — IoU match threshold (higher = stricter)
    false,  // per_class    — track each class independently
    80,     // nr_classes   — number of object classes
    "iou",  // asso_func    — "iou" | "giou" | "diou" | "ciou" | "centroid"
    false   // is_obb       — oriented bounding boxes
);
```

**Paper:** Bewley et al., [*Simple Online and Realtime Tracking*](https://arxiv.org/abs/1602.00763), ICASSP 2016

---

### ByteTrack

**ByteTrack: Multi-Object Tracking by Associating Every Detection Box** — the most widely deployed tracker, striking an excellent balance between speed and accuracy.

**How it works:**
ByteTrack introduces a two-stage association strategy. High-confidence detections (`conf > track_thresh`) are matched first against all active and recently lost tracks. Detections that fall in the low-confidence band (`min_conf < conf < track_thresh`) — which typically correspond to occluded objects — are then matched in a second stage against any unmatched tracked objects. This recovers objects that would otherwise be lost by a hard confidence threshold.

**When to use:**
- General-purpose tracking with a good detector
- Crowded scenes where low-confidence detections carry useful signal
- When you need the best motion-only accuracy/speed tradeoff

```cpp
#include <motcpp/trackers/bytetrack.hpp>

motcpp::trackers::ByteTrack tracker(
    0.3f,   // det_thresh    — global confidence threshold (for new track init)
    30,     // max_age       — frames before a lost track is deleted
    50,     // max_obs       — observation history length
    3,      // min_hits      — hits before a track is confirmed
    0.3f,   // iou_threshold — base IoU matching threshold
    false,  // per_class
    80,     // nr_classes
    "iou",  // asso_func
    false,  // is_obb
    0.1f,   // min_conf      — lower bound of the low-confidence band
    0.45f,  // track_thresh  — splits high vs low confidence detections
    0.8f,   // match_thresh  — first-stage matching threshold
    30,     // track_buffer  — lost track buffer size (frames)
    30.0f   // frame_rate    — used to set buffer_size = frame_rate/30 * track_buffer
);
```

**Paper:** Zhang et al., [*ByteTrack*](https://arxiv.org/abs/2110.06864), ECCV 2022

---

### OC-SORT

**Observation-Centric SORT** — addresses the Kalman filter's drift during occlusions by anchoring motion estimation to actual observations rather than predicted states.

**How it works:**
Standard Kalman trackers accumulate error during occlusion because the filter propagates uncertain predictions. OC-SORT introduces two corrections: (1) **Observation-Centric Momentum (OCM)** — when a track re-matches after occlusion, the velocity is re-estimated from the observation before and after the gap, replacing the stale Kalman velocity. (2) **Observation-Centric Recovery (OCR)** — uses virtual observations interpolated over the occlusion gap to smooth re-initialization. The `delta_t` parameter controls the observation window for momentum re-estimation.

**When to use:**
- Non-linear or erratic motion (pedestrians, sports)
- Scenes with frequent short occlusions
- When object velocity estimates must stay accurate after re-match

```cpp
#include <motcpp/trackers/ocsort.hpp>

motcpp::trackers::OCSort tracker(
    0.2f,    // det_thresh
    30,      // max_age
    50,      // max_obs
    3,       // min_hits
    0.3f,    // iou_threshold
    false,   // per_class
    80,      // nr_classes
    "iou",   // asso_func
    false,   // is_obb
    0.1f,    // min_conf
    3,       // delta_t       — observation window for OCM velocity re-estimation
    0.2f,    // inertia       — weight of previous velocity direction (0 = none)
    false,   // use_byte      — enable ByteTrack-style second association stage
    0.01f,   // Q_xy_scaling  — process noise scaling for position
    0.0001f  // Q_s_scaling   — process noise scaling for scale
);
```

**Paper:** Cao et al., [*Observation-Centric SORT*](https://arxiv.org/abs/2203.14360), CVPR 2023

---

### UCMCTrack

**Uniform Camera Motion Compensation** — projects detections to a ground plane, making tracking robust to camera movement without needing explicit homography estimation.

**How it works:**
Rather than tracking in image coordinates, UCMCTrack maps detections to a ground-plane coordinate system using a simplified camera model. The Kalman filter operates on the `[x, vx, y, vy]` ground-plane state, where uniform camera motion appears as a constant offset that the filter naturally absorbs. This avoids the complex optical-flow or feature-matching required by other CMC approaches. Associations use Mahalanobis distance in ground-plane space, giving principled confidence thresholds (`a1`, `a2`).

**When to use:**
- Drone or aerial footage with moving camera
- Surveillance with pan/tilt cameras
- When ego-motion dominates detection displacement

```cpp
#include <motcpp/trackers/ucmc.hpp>

motcpp::trackers::UCMCTrack tracker(
    0.3f,    // det_thresh
    30,      // max_age
    50,      // max_obs
    3,       // min_hits
    0.3f,    // iou_threshold
    false,   // per_class
    80,      // nr_classes
    "iou",   // asso_func
    false,   // is_obb
    100.0,   // a1           — confirmed-track Mahalanobis gate
    100.0,   // a2           — tentative-track Mahalanobis gate
    5.0,     // wx           — process noise std dev, x-axis
    5.0,     // wy           — process noise std dev, y-axis
    10.0,    // vmax         — maximum expected velocity (ground plane units/frame)
    0.033,   // dt           — time step (set to 1/fps)
    0.5f     // high_score   — confidence split between high/low detection pools
);
```

**Paper:** Yi et al., [*UCMCTrack*](https://arxiv.org/abs/2312.08952), AAAI 2024

---

### OracleTrack

**Motion-Only Tracker with Camera Motion Compensation** — a carefully engineered tracker combining ideas from ByteTrack, OC-SORT, and BoT-SORT into a single high-performance motion-only pipeline.

**How it works:**
OracleTrack layers four techniques:

1. **7D XYAH Kalman filter** with adaptive process noise — noise scales with detection uncertainty rather than being fixed.
2. **ORB-based Camera Motion Compensation (CMC)** — estimates the inter-frame camera warp from ORB feature matches and compensates all track predictions before association, making the tracker robust on moving platforms.
3. **Cascaded ByteTrack association** — first matches high-confidence detections to confirmed tracks using fused score-IoU cost; then matches low-confidence detections to remaining tracked objects.
4. **OC-SORT recovery** — after re-match following occlusion, freezes covariance and applies observation-centric velocity correction to suppress post-occlusion drift.

Track lifecycle: `Tentative → Confirmed → Mature → Lost → Removed`.

**When to use:**
- Moving cameras (drones, vehicles, PTZ) where motion-only is preferred over ReID
- Heavy occlusion environments
- High-speed requirements without sacrificing accuracy

```cpp
#include <motcpp/trackers/oracletrack.hpp>

motcpp::trackers::OracleTrack tracker(
    0.3f,   // det_thresh        — minimum detection confidence
    30,     // max_age           — frames before a lost track is removed
    3,      // min_hits          — frames before a track is confirmed
    9.21f,  // gating_threshold  — Mahalanobis gating chi-squared value (4 DoF → 0.05 p-val)
    4.0f    // max_mahalanobis   — hard distance cap in the cost matrix
);
```

**Performance:** HOTA 66.9 · MOTA 77.3 · IDF1 79.7 · **449 FPS** on MOT17-ablation

---

## ReID-Enhanced Trackers

These trackers incorporate appearance embeddings from a neural Re-Identification (ReID) model. They match tracks not only on IoU but also on visual similarity, enabling recovery after long occlusions and disambiguation of visually similar targets.

### ReID Setup

All ReID trackers accept an ONNX model path:

```bash
# Download OSNet x1.0 (recommended)
wget https://github.com/Geekgineer/motcpp/releases/download/reid-models-v1.0.0/osnet_x1_0.onnx
```

```cpp
// GPU inference (requires ONNX Runtime with CUDA provider)
tracker("osnet_x1_0.onnx", /*use_half=*/false, /*use_gpu=*/true);

// CPU inference
tracker("osnet_x1_0.onnx", /*use_half=*/false, /*use_gpu=*/false);
```

Available ReID models in releases:

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `osnet_x0_25.onnx` | 2 MB | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ |
| `osnet_x0_5.onnx` | 5 MB | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ |
| `osnet_x1_0.onnx` | 16 MB | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ |

---

### DeepOC-SORT

**Deep OC-SORT** — extends OC-SORT with a deep appearance model, providing the occlusion robustness of OC-SORT plus long-range re-identification.

**How it works:**
DeepOC-SORT keeps the full OC-SORT motion pipeline (OCM velocity correction, OCR recovery) and adds an appearance component: each track maintains an EMA-smoothed embedding gallery. The cost matrix fuses IoU distance with cosine embedding distance using a configurable weight `alpha`. When IoU-alone gives an ambiguous match, appearance breaks the tie.

**When to use:**
- Long occlusions where tracks must be re-identified
- Similar-looking targets (e.g., players in sports, uniform crowds)
- When OC-SORT's motion accuracy is needed alongside ReID

```cpp
#include <motcpp/trackers/deepocsort.hpp>

motcpp::trackers::DeepOCSort tracker(
    "osnet_x1_0.onnx",  // reid_weights   — path to ONNX ReID model
    false,              // use_half       — FP16 inference (GPU only)
    false,              // use_gpu        — CUDA inference
    0.2f,               // det_thresh
    30,                 // max_age
    50,                 // max_obs
    3,                  // min_hits
    0.3f,               // iou_threshold
    false,              // per_class
    80,                 // nr_classes
    "iou",              // asso_func
    false,              // is_obb
    0.1f,               // min_conf
    3,                  // delta_t        — OC-SORT OCM window
    0.2f,               // inertia        — OC-SORT velocity dampening
    0.5f                // alpha          — IoU vs appearance blend (0=IoU only, 1=emb only)
);
```

**Paper:** Maggiolino et al., [*Deep OC-SORT*](https://arxiv.org/abs/2302.11813), 2023

---

### StrongSORT

**StrongSORT: Make DeepSORT Great Again** — a heavily improved DeepSORT with Noise-aware State Adaptation (NSA) Kalman filtering and exponential moving average (EMA) appearance updates.

**How it works:**
StrongSORT identifies two weaknesses in DeepSORT: (1) fixed Kalman noise matrices that don't adapt to detection confidence; and (2) single-observation appearance updates that are sensitive to occlusion-frame noise. It fixes both with **NSA Kalman** — measurement noise scales with `1 - confidence`, so low-confidence detections drive smaller corrections — and **EMA appearance** — each track's embedding gallery is updated as a running average, smoothing out occlusion frames. An optional **AFLink** post-processing step further connects fragmented tracklets.

**When to use:**
- High-accuracy benchmarks where HOTA matters more than FPS
- Targets with distinctive appearance that remains consistent
- Offline processing pipelines (AFLink post-processing)

```cpp
#include <motcpp/trackers/strongsort.hpp>

motcpp::trackers::StrongSORT tracker(
    "osnet_x1_0.onnx",  // reid_weights
    false,              // use_half
    false,              // use_gpu
    0.3f,               // det_thresh
    30,                 // max_age
    50,                 // max_obs
    3,                  // min_hits
    0.3f,               // iou_threshold
    false,              // per_class
    80,                 // nr_classes
    "iou",              // asso_func
    false,              // is_obb
    0.1f,               // min_conf
    0.4f,               // ema_alpha      — EMA smoothing for appearance (0=no update, 1=replace)
    0.9f,               // max_dist       — cosine distance gate for ReID matching
    0.3f                // nn_budget      — appearance gallery size per track
);
```

**Paper:** Du et al., [*StrongSORT*](https://arxiv.org/abs/2202.13514), IEEE Transactions on Multimedia 2023

---

### BoT-SORT

**BoT-SORT: Robust Associations Multi-Pedestrian Tracking** — combines camera motion compensation with strong appearance features for robust pedestrian tracking in broadcast and surveillance.

**How it works:**
BoT-SORT extends ByteTrack in two ways. First, it applies **Global Motion Compensation (GMC)** using sparse optical flow or ECC alignment to warp track predictions to the current camera frame before association — this removes apparent object motion caused by camera movement. Second, it fuses IoU cost with ReID cosine distance in a weighted combination. The two-stage ByteTrack association is preserved, so low-confidence detections still participate in a second matching round.

**When to use:**
- Sports broadcasting with moving cameras
- PTZ surveillance cameras
- When both camera motion and appearance matter

```cpp
#include <motcpp/trackers/botsort.hpp>

motcpp::trackers::BotSort tracker(
    "osnet_x1_0.onnx",  // reid_weights
    false,              // use_half
    false,              // use_gpu
    0.3f,               // det_thresh
    30,                 // max_age
    50,                 // max_obs
    3,                  // min_hits
    0.3f,               // iou_threshold
    false,              // per_class
    80,                 // nr_classes
    "iou",              // asso_func
    false,              // is_obb
    0.1f,               // min_conf
    0.45f,              // track_thresh
    0.8f,               // match_thresh
    30,                 // track_buffer
    30.0f,              // frame_rate
    0.5f,               // proximity_thresh — IoU gate before appearance
    0.25f               // appearance_thresh — cosine distance gate
);
```

**Paper:** Aharon et al., [*BoT-SORT*](https://arxiv.org/abs/2206.14651), 2022

---

### HybridSORT

**Hybrid-SORT: Weak Cues Matter for Online Multi-Object Tracking** — integrates height as a proxy for depth into IoU computation, adding a weak but complementary spatial cue that disambiguates detections at similar image-plane locations.

**How it works:**
HybridSORT introduces **Height-Modulated IoU (HMIoU)**: the standard IoU is weighted by the ratio of detection heights, under the assumption that objects at similar depths have similar heights. Objects at different distances (and thus different heights) are penalized even when their projected bounding boxes overlap significantly. On top of this geometric cue, optional ReID embeddings are blended into the cost matrix for long-range re-identification. Association follows the standard two-stage pipeline: high-confidence detections first, then low-confidence.

**When to use:**
- Scenes with depth variation (outdoor pedestrians, vehicles)
- When height provides discriminative signal
- Balanced accuracy and speed with optional ReID

```cpp
#include <motcpp/trackers/hybridsort.hpp>

motcpp::trackers::HybridSort tracker(
    0.3f,    // det_thresh
    30,      // max_age
    50,      // max_obs
    3,       // min_hits
    0.3f,    // iou_threshold
    false,   // per_class
    80,      // nr_classes
    "hmiou", // asso_func   — "iou" (standard) | "hmiou" (height-modulated)
    false,   // is_obb
    0.1f,    // min_conf
    0.45f,   // track_thresh
    0.8f,    // match_thresh
    30,      // track_buffer
    30.0f,   // frame_rate
    1,       // delta_t      — velocity estimation window
    0.1f     // aw_param     — appearance weight in hybrid cost matrix
);
```

**Paper:** Yang et al., [*Hybrid-SORT*](https://arxiv.org/abs/2308.00783), AAAI 2024

---

### BoostTrack

**BoostTrack: Boosting the Similarity Measure and Detection Confidence for Improved Multi-Object Tracking** — current state-of-the-art on MOT17, combining a richer similarity measure with detection confidence boosting.

**How it works:**
BoostTrack attacks two sources of association error. First, it **boosts detection confidence** of boxes that are consistent with existing tracks, reducing false negatives from a detector that under-fires on partially occluded objects. Second, it defines a **combined similarity measure** that jointly considers IoU, Mahalanobis distance, and appearance embedding distance — each normalized and combined with learned weights so no single cue dominates. The tracker also maintains a soft detection buffer: detections just below the threshold are held for one frame to see if they become consistent with a track, enabling recovery from single-frame detection failures.

**When to use:**
- Maximum HOTA/IDF1 on MOT benchmarks
- Scenarios where the detector undershoots on occluded objects
- Research and ablation baselines

```cpp
#include <motcpp/trackers/boosttrack.hpp>

motcpp::trackers::BoostTrackTracker tracker(
    "osnet_x1_0.onnx",  // reid_weights
    false,              // use_half
    false,              // use_gpu
    0.3f,               // det_thresh
    30,                 // max_age
    50,                 // max_obs
    3,                  // min_hits
    0.3f,               // iou_threshold
    false,              // per_class
    80,                 // nr_classes
    "iou",              // asso_func
    false,              // is_obb
    0.1f,               // min_conf
    0.45f,              // track_thresh
    0.8f,               // match_thresh
    30,                 // track_buffer
    30.0f               // frame_rate
);
```

**Performance:** HOTA **67.5** · MOTA **77.1** · IDF1 **79.2** · 75 FPS on MOT17-ablation

**Paper:** Stanojevic et al., [*BoostTrack*](https://arxiv.org/abs/2408.13003), MVA 2024

---

## Selection Guide

```
                        ┌───────────────────┐
                        │  Start here       │
                        └────────┬──────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Need >500 FPS?         │
                    └──┬──────────────────┬───┘
                      Yes                No
                       │                 │
                  ┌────▼────┐   ┌────────▼────────┐
                  │  SORT   │   │ Have ReID model? │
                  │ByteTrack│   └──┬───────────┬───┘
                  └─────────┘     No           Yes
                                   │             │
                         ┌─────────▼──┐   ┌──────▼──────────┐
                         │Moving cam? │   │Need best HOTA?  │
                         └──┬──────┬──┘   └──┬───────────┬───┘
                           Yes    No         Yes          No
                            │      │          │            │
                     ┌──────▼──┐ ┌─▼──────┐ ┌▼──────────┐ ┌▼───────────┐
                     │UCMCTrack│ │OC-SORT │ │BoostTrack │ │StrongSORT  │
                     │OracleTrack ByteTrack │BoT-SORT   │ │HybridSORT  │
                     └─────────┘ └────────┘ └───────────┘ └────────────┘
```

### Quick-reference decision table

| Scenario | Recommended Tracker |
|----------|---------------------|
| Maximum speed, simple scene | SORT |
| General purpose, no ReID | ByteTrack |
| Frequent occlusions, no ReID | OC-SORT |
| Moving / drone camera, no ReID | UCMCTrack · OracleTrack |
| Best motion-only accuracy | OracleTrack |
| Long occlusions, have ReID | DeepOC-SORT · StrongSORT |
| Moving camera + ReID | BoT-SORT |
| Depth variation in scene | HybridSORT |
| Highest benchmark accuracy | BoostTrack |

---

## Common Parameters

All trackers share these base parameters (inherited from `BaseTracker`):

| Parameter | Type | Description |
|-----------|------|-------------|
| `det_thresh` | `float` | Minimum detection confidence to feed into the tracker |
| `max_age` | `int` | Frames a track survives without a match before deletion |
| `max_obs` | `int` | Length of per-track observation history |
| `min_hits` | `int` | Minimum consecutive matches before a track is confirmed |
| `iou_threshold` | `float` | IoU threshold for the association cost matrix gate |
| `per_class` | `bool` | Run independent trackers per object class |
| `nr_classes` | `int` | Number of classes (used when `per_class=true`) |
| `asso_func` | `string` | Cost function: `"iou"` · `"giou"` · `"diou"` · `"ciou"` · `"centroid"` · `"hmiou"` |
| `is_obb` | `bool` | Enable oriented bounding box mode |

## YAML Configuration

All parameters can also be set via YAML config files in `configs/trackers/`:

```yaml
# configs/trackers/bytetrack.yaml
det_thresh: 0.3
max_age: 30
max_obs: 50
min_hits: 3
iou_threshold: 0.3
track_thresh: 0.45
match_thresh: 0.8
track_buffer: 30
frame_rate: 30
```

```cpp
motcpp::trackers::ByteTrack tracker("configs/trackers/bytetrack.yaml");
```
