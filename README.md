<p align="center">
  <img src="docs/images/logo.svg" width="380" alt="motcpp">
</p>

<h1 align="center">motcpp</h1>

<p align="center">
  <strong>Modern C++ Multi-Object Tracking — 10 SOTA algorithms, production-ready, 10–100× faster than Python</strong>
</p>

<p align="center">
  <a href="https://github.com/Geekgineer/motcpp/actions/workflows/ci.yml">
    <img src="https://github.com/Geekgineer/motcpp/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://codecov.io/gh/Geekgineer/motcpp">
    <img src="https://codecov.io/gh/Geekgineer/motcpp/branch/main/graph/badge.svg" alt="Coverage">
  </a>
  <a href="https://github.com/Geekgineer/motcpp/releases">
    <img src="https://img.shields.io/github/v/release/Geekgineer/motcpp" alt="Release">
  </a>
  <a href="https://github.com/Geekgineer/motcpp/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-AGPL--3.0-blue.svg" alt="License">
  </a>
  <a href="https://github.com/Geekgineer/motcpp/stargazers">
    <img src="https://img.shields.io/github/stars/Geekgineer/motcpp?style=social" alt="Stars">
  </a>
</p>

<p align="center">
  <a href="#-features">Features</a> ·
  <a href="#-benchmarks">Benchmarks</a> ·
  <a href="#-installation">Installation</a> ·
  <a href="#-quick-start">Quick Start</a> ·
  <a href="#-trackers">Trackers</a> ·
  <a href="#-documentation">Documentation</a> ·
  <a href="#-citation">Citation</a>
</p>

---

<div align="center">

<table>
<tr>
<td align="center"><img src="docs/images/demo_bytetrack.gif" width="380" alt="ByteTrack"><br><b>ByteTrack</b> — 1100 FPS</td>
<td align="center"><img src="docs/images/demo_ocsort.gif" width="380" alt="OC-SORT"><br><b>OC-SORT</b> — 850 FPS</td>
</tr>
<tr>
<td align="center"><img src="docs/images/demo_boosttrack.gif" width="380" alt="BoostTrack"><br><b>BoostTrack</b> — HOTA 67.5</td>
<td align="center"><img src="docs/images/demo_sort.gif" width="380" alt="SORT"><br><b>SORT</b> — 1250 FPS</td>
</tr>
</table>

</div>

---

**motcpp** is a high-performance C++ library for multi-object tracking (MOT). It implements 10 state-of-the-art algorithms with a unified, modern C++17 API — covering everything from the lightweight SORT baseline to the current BoostTrack SOTA — all ready to drop into production.

Inspired by [BoxMOT](https://github.com/mikel-brostrom/boxmot), motcpp brings the same algorithmic breadth to C++ with a clean CMake integration, ONNX Runtime ReID backend, and built-in MOT benchmark tooling.

## ✨ Features

- **10 SOTA trackers** — SORT, ByteTrack, OC-SORT, DeepOC-SORT, StrongSORT, BoT-SORT, BoostTrack, HybridSORT, UCMCTrack, OracleTrack
- **10–100× faster than Python** — optimized C++ hot paths, zero-copy Eigen matrices
- **Unified API** — one `update(dets, img)` call across all trackers
- **ONNX ReID backend** — plug in any appearance model as an `.onnx` file
- **Camera motion compensation** — ORB/ECC/SoF CMC built-in
- **MOT benchmark tooling** — evaluate on MOT17/MOT20 out of the box
- **>90% test coverage** — GoogleTest suite across all components
- **Cross-platform** — Linux, macOS, Windows (CI-verified)
- **Modern CMake** — `find_package`, `FetchContent`, and vcpkg ready

## 📊 Benchmarks

### MOT17 Ablation Split

Evaluated on the second half of the MOT17 training set using [YOLOX](https://arxiv.org/abs/2107.08430) detections and [FastReID](https://github.com/JDAI-CV/fast-reid) embeddings. Pre-generated data available in [releases](https://github.com/Geekgineer/motcpp/releases). FPS on Intel i9-13900K, single thread.

| Tracker | Type | HOTA ↑ | MOTA ↑ | IDF1 ↑ | FPS ↑ |
|---------|------|:------:|:------:|:------:|:-----:|
| [SORT](https://arxiv.org/abs/1602.00763) | Motion | 62.4 | 75.2 | 69.2 | **1250** |
| [ByteTrack](https://arxiv.org/abs/2110.06864) | Motion | 66.5 | 76.4 | 77.6 | 1100 |
| [OC-SORT](https://arxiv.org/abs/2203.14360) | Motion | 64.6 | 73.9 | 74.4 | 850 |
| [UCMCTrack](https://arxiv.org/abs/2312.08952) | Motion | 64.0 | 75.6 | 73.9 | 980 |
| [OracleTrack](#oracletrack) | Motion | 66.9 | 77.3 | 79.7 | 449 |
| [DeepOC-SORT](https://arxiv.org/abs/2302.11813) | ReID | 65.8 | 75.1 | 76.2 | 120 |
| [StrongSORT](https://arxiv.org/abs/2202.13514) | ReID | 66.2 | 75.8 | 77.1 | 95 |
| [BoT-SORT](https://arxiv.org/abs/2206.14651) | ReID | 66.8 | 76.2 | 78.3 | 85 |
| [HybridSORT](https://arxiv.org/abs/2308.00783) | ReID | 66.4 | 76.0 | 77.8 | 90 |
| [BoostTrack](https://arxiv.org/abs/2408.13003) | ReID | **67.5** | **77.1** | **79.2** | 75 |

### C++ vs Python

| Tracker | C++ (FPS) | Python (FPS) | Speedup |
|---------|:---------:|:------------:|:-------:|
| ByteTrack | 1100 | 45 | **24×** |
| OC-SORT | 850 | 32 | **27×** |
| StrongSORT | 95 | 8 | **12×** |

## 🚀 Installation

### Prerequisites

| Dependency | Version | Required |
|------------|---------|----------|
| C++ compiler (GCC / Clang / MSVC) | C++17 | ✅ |
| CMake | 3.20+ | ✅ |
| OpenCV | 4.x | ✅ |
| Eigen3 | 3.3+ | ✅ |
| yaml-cpp | any | ✅ |
| ONNX Runtime | 1.16+ | ReID only |

<details>
<summary><b>Ubuntu / Debian</b></summary>

```bash
sudo apt-get update && sudo apt-get install -y \
    build-essential cmake \
    libeigen3-dev libopencv-dev libyaml-cpp-dev
```
</details>

<details>
<summary><b>macOS</b></summary>

```bash
brew install cmake eigen opencv yaml-cpp
```
</details>

<details>
<summary><b>Windows (vcpkg)</b></summary>

```powershell
vcpkg install eigen3:x64-windows opencv4:x64-windows yaml-cpp:x64-windows
```
</details>

### Build from Source

```bash
git clone https://github.com/Geekgineer/motcpp.git
cd motcpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
sudo cmake --install build
```

### CMake `find_package`

```cmake
find_package(motcpp REQUIRED)
target_link_libraries(your_target PRIVATE motcpp::motcpp)
```

### CMake `FetchContent`

```cmake
include(FetchContent)
FetchContent_Declare(
    motcpp
    GIT_REPOSITORY https://github.com/Geekgineer/motcpp.git
    GIT_TAG        v1.0.0
)
FetchContent_MakeAvailable(motcpp)
target_link_libraries(your_target PRIVATE motcpp::motcpp)
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `MOTCPP_BUILD_TESTS` | `ON` | GoogleTest unit tests |
| `MOTCPP_BUILD_EXAMPLES` | `ON` | Example binaries |
| `MOTCPP_BUILD_TOOLS` | `ON` | `motcpp_eval` CLI |
| `MOTCPP_ENABLE_ONNX` | `ON` | ONNX Runtime ReID backend |
| `MOTCPP_COVERAGE` | `OFF` | gcov/lcov coverage |
| `BUILD_SHARED_LIBS` | `OFF` | Build as shared library |

## 🎮 Quick Start

### Motion-Only Tracker (no ReID)

```cpp
#include <motcpp/trackers/bytetrack.hpp>
#include <opencv2/opencv.hpp>

int main() {
    motcpp::trackers::ByteTrack tracker;

    cv::VideoCapture cap("video.mp4");
    cv::Mat frame;

    while (cap.read(frame)) {
        // Detector output: [x1, y1, x2, y2, confidence, class_id]
        Eigen::MatrixXf dets = your_detector(frame);

        // Update — returns [x1, y1, x2, y2, track_id, conf, class_id, det_idx]
        Eigen::MatrixXf tracks = tracker.update(dets, frame);

        for (int i = 0; i < tracks.rows(); ++i) {
            int id = static_cast<int>(tracks(i, 4));
            cv::Rect box(tracks(i, 0), tracks(i, 1),
                         tracks(i, 2) - tracks(i, 0),
                         tracks(i, 3) - tracks(i, 1));
            cv::rectangle(frame, box, motcpp::BaseTracker::id_to_color(id), 2);
            cv::putText(frame, "ID " + std::to_string(id),
                        box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        motcpp::BaseTracker::id_to_color(id), 2);
        }

        cv::imshow("Tracking", frame);
        if (cv::waitKey(1) == 27) break;
    }
}
```

### ReID-Enhanced Tracker

```cpp
#include <motcpp/trackers/boosttrack.hpp>

// Point to a ReID ONNX model (see docs/guides/trackers.md for download)
motcpp::trackers::BoostTrackTracker tracker("osnet_x1_0.onnx");

while (cap.read(frame)) {
    Eigen::MatrixXf dets  = detector(frame);
    Eigen::MatrixXf tracks = tracker.update(dets, frame);
    // ...
}
```

### Per-Class Tracking

```cpp
motcpp::trackers::ByteTrack tracker(
    0.3f,   // det_thresh
    30,     // max_age
    50,     // max_obs
    3,      // min_hits
    0.3f,   // iou_threshold
    true,   // per_class — track each class independently
    80      // nr_classes
);
```

### Reset Between Sequences

```cpp
tracker.reset();   // clears all track state and ID counter
```

## 📋 Trackers

### Choosing the Right Tracker

```
Need maximum throughput (>500 FPS)?
    └─ SORT or ByteTrack

General purpose, great accuracy/speed balance?
    └─ ByteTrack · OC-SORT · OracleTrack

Heavy occlusion or non-linear motion?
    └─ OC-SORT · UCMCTrack · OracleTrack

Moving or drone camera?
    └─ UCMCTrack · BoT-SORT · OracleTrack

Have a ReID model and need best accuracy?
    └─ BoostTrack · StrongSORT · BoT-SORT
```

### All Trackers at a Glance

| Tracker | Type | State Space | Key Innovation | Paper |
|---------|------|-------------|----------------|-------|
| **SORT** | Motion | XYSR | IoU + Kalman baseline | [ICASSP 2016](https://arxiv.org/abs/1602.00763) |
| **ByteTrack** | Motion | XYAH | Two-stage low-conf association | [ECCV 2022](https://arxiv.org/abs/2110.06864) |
| **OC-SORT** | Motion | XYSR | Observation-centric momentum | [CVPR 2023](https://arxiv.org/abs/2203.14360) |
| **UCMCTrack** | Motion | Ground plane | Uniform camera motion compensation | [AAAI 2024](https://arxiv.org/abs/2312.08952) |
| **OracleTrack** | Motion | XYAH | CMC + cascaded matching + OC recovery | — |
| **DeepOC-SORT** | ReID | XYSR | OC-SORT + appearance embeddings | [arXiv 2023](https://arxiv.org/abs/2302.11813) |
| **StrongSORT** | ReID | XYAH | NSA Kalman + EMA appearance | [TMM 2023](https://arxiv.org/abs/2202.13514) |
| **BoT-SORT** | ReID | XYSR | GMC camera compensation + ReID | [arXiv 2022](https://arxiv.org/abs/2206.14651) |
| **HybridSORT** | ReID | XYSR | Hybrid IoU + height-modulated IoU | [AAAI 2024](https://arxiv.org/abs/2308.00783) |
| **BoostTrack** | ReID | XYSR | Boosted similarity + detection confidence | [MVA 2024](https://arxiv.org/abs/2408.13003) |

Full parameter reference and per-tracker code snippets: **[docs/guides/trackers.md](docs/guides/trackers.md)**

### ReID Model Download

```bash
# Download OSNet x1.0 (recommended — good accuracy/size balance)
./scripts/auto_benchmark.sh --download-reid

# Or directly:
wget https://github.com/Geekgineer/motcpp/releases/download/reid-models-v1.0.0/osnet_x1_0.onnx
```

## 🧪 Testing

```bash
cmake -B build -DMOTCPP_BUILD_TESTS=ON
cmake --build build -j$(nproc)
cd build && ctest --output-on-failure

# Single test suite
./build/tests/motcpp_tests --gtest_filter=ByteTrackTest.*
```

## 📚 Documentation

| Resource | Description |
|----------|-------------|
| [Getting Started](docs/guides/getting-started.md) | Install, build, and run your first tracker |
| [Tracker Guide](docs/guides/trackers.md) | Full parameter docs for all 10 algorithms |
| [Architecture](docs/guides/architecture.md) | Kalman filters, association, ReID internals |
| [Benchmarking](docs/guides/benchmarking.md) | Reproduce MOT17/20 results |
| [API Reference](docs/api/README.md) | Full C++ API |
| [Tutorials](docs/tutorials/README.md) | Step-by-step examples |

## 🤝 Contributing

Contributions are welcome — bug fixes, new trackers, documentation, benchmarks. See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow.

## 📜 Citation

If you use motcpp in your research, please cite:

```bibtex
@software{motcpp2026,
  author  = {motcpp contributors},
  title   = {motcpp: Modern C++ Multi-Object Tracking Library},
  year    = {2026},
  url     = {https://github.com/Geekgineer/motcpp},
  license = {AGPL-3.0}
}
```

Please also cite the original algorithm papers. See [Acknowledgments](#-acknowledgments) below.

## 📄 License

Licensed under the **GNU Affero General Public License v3.0**. See [LICENSE](LICENSE).

## 🙏 Acknowledgments

### Tracking Algorithms

| Algorithm | Reference |
|-----------|-----------|
| SORT | Bewley et al., [*Simple Online and Realtime Tracking*](https://arxiv.org/abs/1602.00763), ICASSP 2016 |
| ByteTrack | Zhang et al., [*ByteTrack: Multi-Object Tracking by Associating Every Detection Box*](https://arxiv.org/abs/2110.06864), ECCV 2022 |
| OC-SORT | Cao et al., [*Observation-Centric SORT*](https://arxiv.org/abs/2203.14360), CVPR 2023 |
| DeepOC-SORT | Maggiolino et al., [*Deep OC-SORT*](https://arxiv.org/abs/2302.11813), 2023 |
| StrongSORT | Du et al., [*StrongSORT: Make DeepSORT Great Again*](https://arxiv.org/abs/2202.13514), TMM 2023 |
| BoT-SORT | Aharon et al., [*BoT-SORT: Robust Associations Multi-Pedestrian Tracking*](https://arxiv.org/abs/2206.14651), 2022 |
| UCMCTrack | Yi et al., [*UCMCTrack: Multi-Object Tracking with Uniform Camera Motion Compensation*](https://arxiv.org/abs/2312.08952), AAAI 2024 |
| HybridSORT | Yang et al., [*Hybrid-SORT: Weak Cues Matter for Online Multi-Object Tracking*](https://arxiv.org/abs/2308.00783), AAAI 2024 |
| BoostTrack | Stanojevic et al., [*BoostTrack: Boosting the Similarity Measure and Detection Confidence*](https://arxiv.org/abs/2408.13003), MVA 2024 |

### Benchmark Tools

| Resource | Citation |
|----------|---------|
| MOT17 Dataset | Milan et al., [*MOT16: A Benchmark for Multi-Object Tracking*](https://arxiv.org/abs/1603.00831), 2016 |
| YOLOX Detections | Ge et al., [*YOLOX: Exceeding YOLO Series in 2021*](https://arxiv.org/abs/2107.08430), 2021 |
| FastReID Embeddings | He et al., [*FastReID: A Pytorch Toolbox for General Instance Re-identification*](https://github.com/JDAI-CV/fast-reid), 2020 |

### Inspiration

motcpp follows the architecture and algorithmic patterns of **[BoxMOT](https://github.com/mikel-brostrom/boxmot)** by Mikel Broström — the Python multi-object tracking library that inspired this C++ port.

---

<p align="center">Made with ❤️ by the motcpp community</p>
