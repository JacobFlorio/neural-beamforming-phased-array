"""Headline plots for the neural-beamforming study.

Produces:
  - sinr_vs_calerror.png   SINR (median + IQR) vs calibration-error
                           magnitude for each beamformer
  - latency.png            per-call latency for each beamformer
  - beampatterns.png       beampattern magnitude vs azimuth for each
                           method at a single scenario, so you can see
                           what "robust" actually looks like
  - training_curve.png     NN loss and SINR through training
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .array_sim import (
    ArraySpec, calibration_error, simulate_snapshots, steering_vector, true_steering,
)
from .beamformers import cbf, mvdr, mvdr_dl, NNBeamformer


COLORS = {
    "CBF": "#888888",
    "MVDR": "#1f77b4",
    "MVDR-DL": "#2ca02c",
    "NN": "#d62728",
}


def sinr_vs_calerror(out: Path):
    data = json.loads(Path("results/benchmark.json").read_text())
    mags = data["cal_mags"]
    methods = list(data["per_mag"][str(mags[0])].keys())

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in methods:
        med = [data["per_mag"][str(mag)][m]["median"] for mag in mags]
        q25 = [data["per_mag"][str(mag)][m]["q25"] for mag in mags]
        q75 = [data["per_mag"][str(mag)][m]["q75"] for mag in mags]
        ax.fill_between(mags, q25, q75, color=COLORS.get(m, "#444"), alpha=0.20)
        ax.plot(mags, med, "o-", color=COLORS.get(m, "#444"), lw=2.5, markersize=7,
                label=m)

    ax.set_xlabel("calibration error magnitude (per-element gain/phase std)")
    ax.set_ylabel("out-of-sample SINR [dB]")
    ax.set_title("SINR vs calibration error — robustness sweep (median + IQR, n=200)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved → {out}")


def latency_plot(out: Path):
    data = json.loads(Path("results/benchmark.json").read_text())
    lat = data["latency_us"]
    methods = list(lat.keys())
    values = [lat[m] for m in methods]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(methods, values,
                  color=[COLORS.get(m, "#444") for m in methods],
                  edgecolor="#222")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f} μs",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("latency per weight computation [μs]")
    ax.set_title("Latency — one weight update per call, median of 500 runs")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved → {out}")


def beampatterns(out: Path, nn_path: Path = Path("results/nn_beamformer.pt"),
                  cal_mag: float = 0.05):
    """Plot |w^H a(θ)|² vs θ for each method on a fixed scenario."""
    spec = ArraySpec(n_elem=16)
    gen = torch.Generator().manual_seed(7)
    cal_err = calibration_error(spec, cal_mag, generator=gen)
    sig_theta = 0.0
    interferers = [(0.35, 18), (-0.5, 15), (0.8, 12)]
    _, R, _, a_nom = simulate_snapshots(
        spec, 256, sig_theta, interferers, 10.0, cal_err, generator=gen,
    )

    w_cbf = cbf(R, a_nom)
    w_mvdr = mvdr(R, a_nom)
    w_dl = mvdr_dl(R, a_nom, load_frac=0.3)

    nn = None
    if nn_path.exists():
        blob = torch.load(nn_path, map_location="cpu", weights_only=False)
        nn = NNBeamformer(**blob["config"])
        nn.load_state_dict(blob["state_dict"])
        nn.eval()
    with torch.no_grad():
        w_nn = nn(R, a_nom) if nn is not None else None

    thetas = np.linspace(-np.pi / 2, np.pi / 2, 401)
    t_t = torch.tensor(thetas, dtype=torch.float64)
    A = torch.zeros(spec.n_elem, len(thetas), dtype=torch.complex128)
    for i, theta in enumerate(thetas):
        A[:, i] = true_steering(spec, torch.tensor(theta, dtype=torch.float64), cal_err)

    def response_db(w):
        g = (w.conj() @ A).abs()
        g = g / g.max()
        return 20 * torch.log10(g.clamp(min=1e-6)).numpy()

    fig, ax = plt.subplots(figsize=(9, 5))
    theta_deg = np.degrees(thetas)

    for name, w in [("CBF", w_cbf), ("MVDR", w_mvdr), ("MVDR-DL", w_dl)]:
        ax.plot(theta_deg, response_db(w), color=COLORS[name], lw=2, label=name)
    if w_nn is not None:
        ax.plot(theta_deg, response_db(w_nn), color=COLORS["NN"], lw=2, label="NN")

    ax.axvline(np.degrees(sig_theta), color="black", ls=":", lw=1, alpha=0.7,
               label=f"signal @ {np.degrees(sig_theta):.0f}°")
    for (th, _) in interferers:
        ax.axvline(np.degrees(th), color="#d62728", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("azimuth [deg]")
    ax.set_ylabel("normalized response [dB]")
    ax.set_ylim(-60, 3)
    ax.set_title(f"Beampatterns at {int(cal_mag*100)}% calibration error "
                 "(dashed red = interferers)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved → {out}")


def training_curve(out: Path):
    blob = torch.load("results/nn_beamformer.pt", map_location="cpu", weights_only=False)
    hist = blob["history"]
    steps = [h["step"] for h in hist]
    loss = [h["loss"] for h in hist]
    sinr = [h["sinr_db"] for h in hist]

    # Moving average with np.convolve — keeps lengths aligned
    wsize = 100
    def ma(x):
        a = np.array(x, dtype=float)
        kernel = np.ones(wsize) / wsize
        smooth = np.convolve(a, kernel, mode="valid")
        ma_steps = np.array(steps[wsize - 1:])
        return ma_steps, smooth

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax_l, ax_s = axes

    ax_l.plot(steps, loss, color="#ccc", lw=0.5, alpha=0.6)
    if len(loss) > wsize:
        xs, ys = ma(loss)
        ax_l.plot(xs, ys, color="#d62728", lw=2, label=f"ma({wsize})")
    ax_l.set_xlabel("step")
    ax_l.set_ylabel("-log(SINR) loss")
    ax_l.set_title("Training loss")
    ax_l.grid(alpha=0.3)

    ax_s.plot(steps, sinr, color="#ccc", lw=0.5, alpha=0.6)
    if len(sinr) > wsize:
        xs, ys = ma(sinr)
        ax_s.plot(xs, ys, color="#d62728", lw=2, label=f"ma({wsize})")
    ax_s.set_xlabel("step")
    ax_s.set_ylabel("out-of-sample SINR [dB]")
    ax_s.set_title("NN SINR on random training scenarios")
    ax_s.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved → {out}")


def main():
    sinr_vs_calerror(Path("results/sinr_vs_calerror.png"))
    latency_plot(Path("results/latency.png"))
    beampatterns(Path("results/beampatterns.png"))
    training_curve(Path("results/training_curve.png"))


if __name__ == "__main__":
    main()
