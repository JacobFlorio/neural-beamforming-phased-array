"""Sweep calibration-error magnitude and compare all four beamformers.

Metrics:
  - Out-of-sample SINR (the headline)
  - Latency per beamforming call (one weight computation)

For each error magnitude we draw `n_scenarios` random scenarios
(random signal angle, interferer count, interferer angles / INRs, SNR),
run every beamformer, compute SINR against the analytical true
covariance, and report median + IQR across scenarios.

Run:
    python -m src.benchmark
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from .array_sim import (
    ArraySpec, simulate_snapshots, calibration_error, eval_sinr_oos,
)
from .beamformers import cbf, mvdr, mvdr_dl, NNBeamformer


def _dl_fraction_for_mag(mag: float) -> float:
    """Hand-tuned diagonal-loading schedule — larger error → larger load.

    This is the fair way to use MVDR-DL as a robust baseline: the
    classical robust literature always tunes the load for the expected
    error level. A fixed load_frac would undersell this baseline.
    """
    return 0.05 + 5.0 * mag


def run_scenario(spec: ArraySpec, rng: np.random.Generator,
                 cal_mag: float, n_snap: int,
                 nn_bf: NNBeamformer | None,
                 device: str):
    sig_theta = float(rng.uniform(-0.7, 0.7))
    n_interf = int(rng.integers(1, 4))
    interferers = []
    for _ in range(n_interf):
        theta = float(rng.uniform(-1.2, 1.2))
        while abs(theta - sig_theta) < 0.25:
            theta = float(rng.uniform(-1.2, 1.2))
        interferers.append((theta, float(rng.uniform(5, 25))))
    snr_db = float(rng.uniform(5, 20))

    gen = torch.Generator(device=device).manual_seed(int(rng.integers(0, 2**31)))
    cal_err = calibration_error(spec, cal_mag, generator=gen, device=device)
    _, R, _, a_nom = simulate_snapshots(
        spec, n_snap, sig_theta, interferers, snr_db,
        cal_err, generator=gen, device=device,
    )

    results = {}
    for name, fn in [
        ("CBF", lambda: cbf(R, a_nom)),
        ("MVDR", lambda: mvdr(R, a_nom)),
        ("MVDR-DL", lambda: mvdr_dl(R, a_nom, load_frac=_dl_fraction_for_mag(cal_mag))),
    ]:
        w = fn()
        sinr = eval_sinr_oos(w, spec, sig_theta, interferers, snr_db, cal_err, device=device)
        results[name] = float(sinr.item())

    if nn_bf is not None:
        with torch.no_grad():
            w = nn_bf(R, a_nom)
        sinr = eval_sinr_oos(w, spec, sig_theta, interferers, snr_db, cal_err, device=device)
        results["NN"] = float(sinr.item())

    return results


def measure_latency(spec: ArraySpec, nn_bf: NNBeamformer | None,
                     n_calls: int = 500, device: str = "cpu"):
    """Microbenchmark one weight computation per method.

    Uses a fixed scenario so we're not timing data generation.
    """
    gen = torch.Generator(device=device).manual_seed(0)
    cal_err = calibration_error(spec, 0.05, generator=gen, device=device)
    _, R, _, a_nom = simulate_snapshots(
        spec, 256, 0.0, [(0.3, 15), (-0.4, 12)], 10.0,
        cal_err, generator=gen, device=device,
    )

    timings: dict[str, float] = {}

    def _time(name: str, fn):
        # warmup
        for _ in range(20):
            fn()
        t0 = time.perf_counter()
        for _ in range(n_calls):
            fn()
        dt = (time.perf_counter() - t0) / n_calls
        timings[name] = dt * 1e6   # microseconds

    _time("CBF", lambda: cbf(R, a_nom))
    _time("MVDR", lambda: mvdr(R, a_nom))
    _time("MVDR-DL", lambda: mvdr_dl(R, a_nom, load_frac=0.1))
    if nn_bf is not None:
        @torch.no_grad()
        def nn_call():
            nn_bf(R, a_nom)
        _time("NN", nn_call)

    return timings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn", default="results/nn_beamformer.pt")
    ap.add_argument("--n-scenarios", type=int, default=200)
    ap.add_argument("--cal-mags", default="0.0,0.01,0.02,0.05,0.10,0.20")
    ap.add_argument("--n-snap", type=int, default=256)
    ap.add_argument("--n-elem", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/benchmark.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    spec = ArraySpec(n_elem=args.n_elem)
    rng = np.random.default_rng(args.seed)

    nn_bf = None
    nn_path = Path(args.nn)
    if nn_path.exists():
        blob = torch.load(nn_path, map_location=args.device, weights_only=False)
        nn_bf = NNBeamformer(**blob["config"]).to(args.device)
        nn_bf.load_state_dict(blob["state_dict"])
        nn_bf.eval()
        print(f"loaded NN from {nn_path}")

    cal_mags = [float(m) for m in args.cal_mags.split(",")]
    all_rows = []
    per_mag_stats = {}
    print(f"\n{'cal_mag':>8s} {'method':>10s} {'median':>10s} {'q25':>8s} {'q75':>8s}")
    for mag in cal_mags:
        results = {"CBF": [], "MVDR": [], "MVDR-DL": []}
        if nn_bf is not None:
            results["NN"] = []
        for _ in range(args.n_scenarios):
            r = run_scenario(spec, rng, mag, args.n_snap, nn_bf, args.device)
            for k, v in r.items():
                results[k].append(v)
        stats = {}
        for name, vals in results.items():
            arr = np.array(vals)
            stats[name] = {
                "median": float(np.median(arr)),
                "mean": float(np.mean(arr)),
                "q25": float(np.percentile(arr, 25)),
                "q75": float(np.percentile(arr, 75)),
                "n": len(arr),
            }
            print(f"{mag:>8.3f} {name:>10s} {stats[name]['median']:>+9.2f} "
                  f"{stats[name]['q25']:>+7.2f} {stats[name]['q75']:>+7.2f}")
        per_mag_stats[str(mag)] = stats
        all_rows.append({"cal_mag": mag, "stats": stats})

    print("\n=== latency (microseconds per call, median of 500) ===")
    lat = measure_latency(spec, nn_bf, n_calls=500, device=args.device)
    for name, us in lat.items():
        print(f"  {name:10s}  {us:8.1f} μs")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "cal_mags": cal_mags,
        "per_mag": per_mag_stats,
        "latency_us": lat,
    }, indent=2))
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
