# Neural Beamforming for Phased Arrays

Learned beamformer for a 16-element mmWave phased array, benchmarked against classical MVDR (Capon) and MUSIC.

## Research question
Can a small neural beamformer match MVDR interference-rejection performance while running 10× faster on an embedded GPU, and how does it generalize to off-distribution array calibration errors?

## Approach
1. Generate synthetic array data with known DoA + interferers.
2. Train a complex-valued MLP to map covariance → weights.
3. Compare against MVDR, MUSIC, and ESPRIT baselines.
4. Inject random gain/phase mismatches to test robustness.
5. Deploy to Jetson Orin Nano and measure latency.

## Deliverables
- Data generation + training in `src/`
- SINR-vs-SNR curves in `results/`
- Latency benchmarks in `results/latency.md`
