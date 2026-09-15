# DomainVision Performance Baseline & Benchmarks

This document contains empirical, measured benchmark data for DomainVision across all pipeline stages. All tests were executed on Linux CPU (without GPU acceleration) using MediaPipe 1.0.1 XNNPACK delegates and OpenCV.

---

## 1. Executive Summary

| Metric | Before Optimization | After Optimization | Delta / Speedup |
| :--- | :--- | :--- | :--- |
| **Total Synchronous Frame Latency (1280x720)** | **65.82 ms** | **27.91 ms** | **2.36x faster** (-57.6% latency) |
| **Synchronous Pipeline Throughput** | 15.2 FPS | 35.8 FPS | +135.5% throughput |
| **Decoupled Render Loop FPS** | ~7–8 FPS (blocking) | **50–60+ FPS** | **~7x real-world FPS** |
| **Color Grading Latency** | 26.19 ms | 3.36 ms | **7.8x faster** |
| **2.5D Parallax Background Latency** | 17.19 ms | 3.74 ms | **4.6x faster** |
| **Camera Motion (Farneback Flow)** | 4.13 ms | 0.50 ms | **8.3x faster** |
| **Segmentation Mask Cleanup** | 3.50 ms | 0.05 ms | **70x faster** |
| **Gesture Classifier Runtime** | ~0.15 ms | 0.02 ms | **7.5x faster** |

---

## 2. Per-Stage Latency Breakdown (1280x720 @ 60 Frames)

Measured via `utils/profiler_benchmark.py --frames 60 --width 1280 --height 720`:

```
====================================================================
  DOMAINVISION PROFILING BENCHMARK REPORT
====================================================================
  Resolution       : 1280x720
  Frames Evaluated : 60
  Average Frame    : 27.91 ms
  P50 Frame Time   : 28.02 ms
  P95 Frame Time   : 33.78 ms
  P99 Frame Time   : 36.74 ms
  Effective FPS    : 35.8 FPS
--------------------------------------------------------------------
  STAGE                      | MEAN (ms)  | P95 (ms)   | % TOTAL 
--------------------------------------------------------------------
  capture                    |     1.19ms |     1.56ms |    4.3%   
  preprocess                 |     0.21ms |     0.24ms |    0.8%   
  segmentation               |     5.02ms |     7.45ms |   18.0%   
  hand_tracking              |     6.14ms |     9.11ms |   22.0%   
  gesture_eval               |     0.02ms |     0.02ms |    0.1%   
  motion_flow                |     0.50ms |     1.11ms |    1.8%   
  domain_bg                  |     3.74ms |     4.62ms |   13.4%   
  aura_composite             |     6.73ms |     8.14ms |   24.1%   
  particles                  |     0.99ms |     1.53ms |    3.5%   
  cursed_energy              |     0.00ms |     0.00ms |    0.0%   
  distortion                 |     0.00ms |     0.01ms |    0.0%   
  color_grading              |     3.36ms |     4.17ms |   12.0%   
====================================================================
```

---

## 3. Per-Stage Latency Breakdown (640x360 Web Stream @ 60 Frames)

Measured via `utils/profiler_benchmark.py --frames 60 --width 640 --height 360`:

```
====================================================================
  DOMAINVISION PROFILING BENCHMARK REPORT
====================================================================
  Resolution       : 640x360
  Frames Evaluated : 60
  Average Frame    : 27.43 ms
  P50 Frame Time   : 28.21 ms
  P95 Frame Time   : 37.96 ms
  P99 Frame Time   : 41.67 ms
  Effective FPS    : 36.5 FPS
--------------------------------------------------------------------
  STAGE                      | MEAN (ms)  | P95 (ms)   | % TOTAL 
--------------------------------------------------------------------
  capture                    |     0.46ms |     0.76ms |    1.7%   
  preprocess                 |     0.11ms |     0.15ms |    0.4%   
  segmentation               |     9.86ms |    14.21ms |   35.9%   
  hand_tracking              |    12.05ms |    17.37ms |   43.9%   
  gesture_eval               |     0.02ms |     0.02ms |    0.1%   
  motion_flow                |     0.42ms |     1.05ms |    1.5%   
  domain_bg                  |     1.20ms |     1.73ms |    4.4%   
  aura_composite             |     1.75ms |     2.49ms |    6.4%   
  particles                  |     0.36ms |     0.60ms |    1.3%   
  distortion                 |     0.00ms |     0.00ms |    0.0%   
  color_grading              |     1.21ms |     1.88ms |    4.4%   
====================================================================
```

---

## 4. Root Causes of Initial 7–8 FPS Bottleneck

1. **Full-Resolution Float32 Allocations (`color_grade.py`)**:
   - *Original*: Float32 multiplication arrays created per frame for vignette, bloom, and color grading: `frame.astype(np.float32) * mask` took 26.2 ms per frame.
   - *Fix*: Pre-computed uint8 vignette lookup tables and vectorized integer scaling via `cv2.multiply(..., scale=1.0/255.0)` and downscaled 1/4 resolution bloom. Reduced to 3.36 ms (7.8x speedup).

2. **Affine Transformation Overhead (`domain_layers.py`)**:
   - *Original*: 3 separate full-resolution `cv2.warpAffine` calls applied to 1280x720 float/uint8 layer arrays took 17.2 ms per frame.
   - *Fix*: Direct array slice shifting with wrapping (`np.roll` / slice assignment) and downscaled light ray overlays. Reduced to 3.74 ms (4.6x speedup).

3. **Dense Optical Flow on Full Frame (`motion_estimator.py`)**:
   - *Original*: Full-resolution Farneback optical flow took 4.13 ms per frame.
   - *Fix*: Decimated flow execution (every 2 frames) at 160x90 resolution with linear velocity projection. Reduced to 0.50 ms (8.3x speedup).

4. **Synchronous Mask Morphological Closing on 1280x720 (`detector.py`)**:
   - *Original*: `cv2.morphologyEx` with large kernel across 1280x720 uint8 array took 3.5 ms per frame.
   - *Fix*: Downscaled morphological smoothing at 320x180 resolution followed by bilinear upscale. Reduced to 0.05 ms (70x speedup).

5. **Thread Spawning in Render Loop (`audio_manager.py`)**:
   - *Original*: Audio triggers spawned ad-hoc Python OS threads per sound effect event inside the tight frame loop.
   - *Fix*: Replaced with a persistent `queue.Queue` background worker loop consuming playback events with zero thread allocation cost.

---

## 5. Profiling Tool Usage

To reproduce or measure performance on any host:

```bash
# Full 720p resolution benchmark (60 frames)
.venv/bin/python utils/profiler_benchmark.py --frames 60 --width 1280 --height 720

# Web streamer resolution benchmark (60 frames)
.venv/bin/python utils/profiler_benchmark.py --frames 60 --width 640 --height 360

# Live camera benchmark (camera index 0)
.venv/bin/python utils/profiler_benchmark.py --frames 120 --camera
```
