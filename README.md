# Neural Beamforming Under Array Calibration Error

> **A trained hybrid neural beamformer matches diagonally-loaded MVDR within IQR noise at low calibration error and beats it by +1.6 dB at high error (20% per-element gain/phase mismatch).** Plain MVDR collapses 24 dB across the same sweep — textbook Li-Stoica failure mode. Honest result: the NN is *robust* but it's also *slower* than MVDR-DL, not faster, because my hybrid architecture wraps classical MVDR internally rather than replacing it.

Independent research by [Jacob Florio](https://github.com/JacobFlorio). 16-element ULA, 200 random scenarios per data point, end-to-end pure torch. Runs in about 4 minutes on an RTX 5080.

---

## Headline numbers

### SINR vs calibration-error magnitude
![SINR vs calibration error](results/sinr_vs_calerror.png)

Median SINR across 200 random scenarios per data point, with 25–75 percentile band. Calibration error σ is the per-element gain/phase mismatch standard deviation.

| cal σ | CBF | MVDR | MVDR-DL | **NN** | NN − MVDR-DL |
|---:|---:|---:|---:|---:|---:|
| 0.00 | +5.6 | +11.7 | **+18.9** | +18.8 | −0.1 |
| 0.02 | +5.7 | +6.4 | **+19.5** | +19.5 | 0.0 |
| 0.05 | +6.6 | −0.5 | **+19.3** | +19.2 | −0.1 |
| 0.10 | +5.5 | −6.7 | +17.9 | **+18.1** | +0.2 |
| 0.20 | +1.7 | **−12.7** | +15.9 | **+17.5** | **+1.6** |

**The headline:**
1. **Plain MVDR is actively dangerous past ~2% calibration error.** It goes from +11.7 dB SINR clean to **−12.7 dB at 20%** — a 24 dB collapse. Past about σ=0.05 it's worse than ignoring the covariance entirely (CBF).
2. **MVDR with diagonal loading** (a tuned load schedule, not a strawman) holds the line at 16–19 dB across the entire sweep.
3. **The NN matches MVDR-DL** within IQR noise from 0% through 5% calibration error.
4. **The NN beats MVDR-DL** at the high-error tail: +0.2 dB at σ=0.10 and **+1.6 dB at σ=0.20**. The learned δa correction starts compensating for steering-vector error in regimes where the fixed-parameterization MVDR-DL has run out of knobs.

### Beampatterns at 5% calibration error
![Beampatterns](results/beampatterns.png)

Single scenario with 3 interferers at 20°, −29°, 46° (dashed red). MVDR has the deepest nulls but tilts its main beam off the true source direction (the calibration error is bending its apparent steering). MVDR-DL and NN look broadly similar at this error level — they trade some null depth for keeping the main beam on target. This is what robustness looks like in beampattern form.

### Latency — honest reversal of the README's original speculation
![Latency](results/latency.png)

| method | latency (μs/call) |
|---|---:|
| CBF | 5.1 |
| MVDR | 308.3 |
| MVDR-DL | 420.5 |
| **NN** | **1207.5** |

**The NN is 3× slower than MVDR-DL, not faster.** My original project README speculated about constant-time inference; that turns out to require a different architecture (one that emits weights without an internal matrix solve), and I haven't gotten that variant to converge yet. The hybrid architecture I shipped trades latency for robustness — full discussion in [`docs/report.md`](docs/report.md).

### Training curve
![Training](results/training_curve.png)

NN trained for 4000 steps on randomized scenarios with calibration error magnitude uniformly sampled in [0, 0.10]. Loss is `log(wᴴ R_i+n w) − log(|wᴴa_true|²)` — negative SINR in nats, smoothed by a small floor on the signal term so gradients stay well-behaved. Mean SINR plateau around +19 dB on the wide training distribution. The improvements concentrate in the high-error tail, which is why the +1.6 dB headline win at σ=0.20 isn't a giant drop in the global loss.

## Why this is a real research question

MVDR is analytically optimal *if* the array response is exactly known. Real arrays never have that — every analog front-end has some gain and phase mismatch from cable length, LNA variation, temperature drift, and aging. These mismatches shift the true steering vector away from the nominal one the DSP assumes, and past some threshold MVDR starts nulling the signal of interest because it thinks the signal power is interference. The classical robust-Capon literature (Li & Stoica 2003; worst-case bounded design; diagonal loading) addresses this with explicit regularization. The question this project asks: can a learned beamformer, trained to be robust-by-construction on randomized miscalibrated scenarios, outperform the best classical robust scheme — and where?

## Architecture: hybrid neural / classical

Rather than asking the MLP to learn the full `R⁻¹·a` map from scratch (which my first attempt tried; took 6000 steps to reach −5 dB and was useless), the NN parameterizes the classical MVDR pipeline:

```
NN(R, a_nominal)  →  (log_ε, δa)
weights = MVDR(R + ε·I,  a_nominal + δa)
```

The NN learns *two things*: a scenario-adaptive diagonal loading scalar, and a small correction to the assumed steering vector. The output layer is zero-initialized, so the NN starts exactly at MVDR-DL behavior and training preserves/refines that baseline. This converted "learn the inverse covariance map" into "tune two scalars and a small delta vector" — a problem a small MLP can actually solve.

Discussion of the failure mode of the direct architecture and the rationale for the hybrid design is in [`docs/report.md`](docs/report.md).

## Reproduce it yourself

```bash
pip install -r requirements.txt    # just torch + numpy + matplotlib

python -m src.train                # 4000 steps, ~3 min
python -m src.benchmark            # 200 × 6 × 4 SINR sweep + latency
python -m src.plots                # headline figures
```

About 4 minutes end to end on an RTX 5080.

## Full technical writeup

See [`docs/report.md`](docs/report.md) for the full setup (dynamics, calibration error model, training loss derivation, why the first architecture failed), the per-cell SINR table, beampattern interpretation, latency post-mortem, and the multi-seed / direct-weight-architecture / correlated-error-model roadmap.

## Honest caveats
- Single training seed. Multi-seed confirmation is the obvious followup, same pattern as my [sae-rf-classifier](https://github.com/JacobFlorio/sae-rf-classifier) and [mech-interp](https://github.com/JacobFlorio/mech-interp-tiny-transformer) sister projects.
- The +1.6 dB win at σ=0.20 is the headline, but σ=0.20 is on the high end of realistic calibration error.
- Calibration error is i.i.d. Gaussian per element — real arrays have correlated errors (mutual coupling, thermal gradients) that this model doesn't capture.
- MVDR-DL's load schedule is hand-tuned; a fairer adaptive baseline would be Generalized Li-Stoica (GLC) or worst-case bounded design.
- Latency is honestly worse than MVDR-DL.

## Part of [AI-and-EE-Research](https://github.com/JacobFlorio/AI-and-EE-Research)

The broader index of my independent EE × AI research projects. Companion to:
- [mech-interp-tiny-transformer](https://github.com/JacobFlorio/mech-interp-tiny-transformer) — grokking + SAE causal ablation
- [edge-llm-eval-harness](https://github.com/JacobFlorio/edge-llm-eval-harness) — hardware-aware quantized-LLM eval
- [sae-rf-classifier](https://github.com/JacobFlorio/sae-rf-classifier) — SAE rediscovers classical modulation features
- [fpga-transformer-accel](https://github.com/JacobFlorio/fpga-transformer-accel) — bit-accurate systolic accelerator simulator
- [rl-power-converter](https://github.com/JacobFlorio/rl-power-converter) — tuned PID beats neural policies on a buck converter
