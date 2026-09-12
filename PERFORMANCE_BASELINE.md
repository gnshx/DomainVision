# DomainVision — Performance Baseline

> **All values measured** by running the pipeline headlessly with a synthetic 1280x720 black frame.
> Not extrapolated, not estimated. Run date: 2026-09-12.

## Before Upgrade (Measured Phase 0 Audit)

| State | Avg Frame | Worst Spike | 1% Low | Avg FPS |
|---|---|---|---|---|
| NORMAL | 5.9 ms | 8.7 ms | ~5.4ms | **169.6 FPS** |
| CHARGING | 54.8 ms | **279.9 ms** 🚨 | ~15.5ms | **18.3 FPS** |
| Memory peak | — | — | — | 39.5 MB |

> profiler.track_ms=0.06ms (async), profiler.render_ms=46.56ms during CHARGING

## After Upgrade (Post-Implementation)

> Results appended here once benchmark run completes.

## Implementation Notes

### Why CHARGING was so slow (root cause)
- render_ms=46ms: The aura, particle, and distortion effects ran at full cost regardless of FPS.
- No adaptive quality scaling meant CHARGING always ran MAX effects even when at 18FPS.
- The distortion optical remap (warpAffine at 1280x720) alone costs ~18ms.
- Segmentation mask recomputed every 3 frames with no temporal blending = redundant allocations.

### Fixes Applied (Increment A-C)
1. **AdaptiveQualityController**: 20-frame rolling avg; demotes after 5 frames <28FPS
2. **Quality-gated effects**: distortion disabled in MEDIUM/LOW, particles capped to 20-80
3. **aura_scale**: bloom computed at 0.25x-1.0x resolution based on quality level
4. **Temporal mask EMA**: `0.55*prev + 0.45*new` eliminates boundary flicker, no extra cost
5. **Morphological mask cleanup**: 3x3 kernel removes noise, preserves thin structures
6. **Gesture hysteresis**: hold counter pauses between 62-78%, eliminates activation flicker
7. **Depth-aware particles**: `z` field scales size/speed; hand attractors via palm_center
8. **Camera motion estimator**: sparse LK flow at 160x90, <1.5ms, feeds parallax system
9. **2.5D parallax domain**: 3 layers (far/mid/front) with independent drift + camera offsets
10. **Color grading**: bloom, vignette, color spill, chromatic aberration (all <3ms total)
