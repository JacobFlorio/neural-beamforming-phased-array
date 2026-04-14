"""Train the NN beamformer by maximizing out-of-sample SINR through the
differentiable sim.

Each training step samples a random scenario (signal angle, interferer
count/angles/INRs, SNR, calibration-error realization), simulates a
sample covariance R from a finite snapshot draw, pushes (R, a_nominal)
through the NN to get weights w, then computes the analytical
out-of-sample SINR against the *true* miscalibrated system and uses
-SINR as the loss. Gradients flow all the way back to the network.

Domain randomization over calibration-error magnitude: each batch
draws a calibration error at a uniformly random magnitude in
[0, cal_max]. This gives a single robust policy over the whole error
range, rather than training a separate NN per magnitude.

Run:
    python -m src.train
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .array_sim import (
    ArraySpec, simulate_snapshots, calibration_error,
    eval_sinr_oos, r_interference_plus_noise, true_steering, sinr_from_weights,
    steering_vector,
)
from .beamformers import NNBeamformer


def oracle_weights(a_true: torch.Tensor, R_in: torch.Tensor) -> torch.Tensor:
    """MVDR weights computed with the ground-truth steering vector and the
    analytical interference+noise covariance. This is the "oracle" we
    teach the NN to imitate — a robust-Capon target that the NN has to
    recover using only its nominal-steering and sample-R inputs."""
    R_reg = R_in + 1e-9 * torch.eye(R_in.shape[0], dtype=R_in.dtype, device=R_in.device)
    Rinv_a = torch.linalg.solve(R_reg, a_true)
    denom = (a_true.conj() @ Rinv_a).real
    return Rinv_a / denom


def random_scenario(spec: ArraySpec, rng: np.random.Generator,
                     cal_max: float, n_snap: int,
                     device: str = "cpu"):
    """Draw a random beamforming scenario."""
    sig_theta = float(rng.uniform(-0.7, 0.7))    # radians
    n_interf = int(rng.integers(1, 4))            # 1-3 interferers
    interferers = []
    for _ in range(n_interf):
        theta = float(rng.uniform(-1.2, 1.2))
        # keep interferers at least ~15 degrees away from signal
        while abs(theta - sig_theta) < 0.25:
            theta = float(rng.uniform(-1.2, 1.2))
        inr_db = float(rng.uniform(5, 25))
        interferers.append((theta, inr_db))
    snr_db = float(rng.uniform(5, 20))
    cal_mag = float(rng.uniform(0.0, cal_max))
    gen = torch.Generator(device=device).manual_seed(int(rng.integers(0, 2 ** 31)))
    cal_err = calibration_error(spec, cal_mag, generator=gen, device=device)
    _, R, _, a_nominal = simulate_snapshots(
        spec, n_snap, sig_theta, interferers, snr_db, cal_err,
        generator=gen, device=device,
    )
    return {
        "sig_theta": sig_theta,
        "interferers": interferers,
        "snr_db": snr_db,
        "cal_err": cal_err,
        "R": R,
        "a_nominal": a_nominal,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-elem", type=int, default=16)
    ap.add_argument("--n-snap", type=int, default=256)
    ap.add_argument("--cal-max", type=float, default=0.10)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/nn_beamformer.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    spec = ArraySpec(n_elem=args.n_elem)
    net = NNBeamformer(n_elem=args.n_elem, hidden=args.hidden).to(args.device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    history = []
    for step in range(args.steps + 1):
        scenario = random_scenario(spec, rng, cal_max=args.cal_max,
                                    n_snap=args.n_snap, device=args.device)
        R = scenario["R"]
        a_nom = scenario["a_nominal"]
        cal_err = scenario["cal_err"]

        # True steering and analytical interference+noise covariance
        a_true = true_steering(spec, torch.tensor(scenario["sig_theta"],
                                                   dtype=torch.float64,
                                                   device=args.device),
                                cal_err)
        R_in = r_interference_plus_noise(spec, scenario["interferers"],
                                          scenario["snr_db"], cal_err,
                                          device=args.device)

        w = net(R, a_nom)

        # Loss: minimize log(interference+noise power / signal power).
        # This is -SINR in nats, and unlike the raw dB SINR it's smooth
        # everywhere (the signal floor prevents the log from going to -∞
        # when the NN briefly nulls the signal).
        wRw = (w.conj() @ R_in @ w).real
        wa = (w.conj() * a_true).sum(dim=-1)
        signal = (wa.conj() * wa).real
        loss = torch.log(wRw + 1e-12) - torch.log(signal + 1e-6)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        opt.step()

        # Monitoring: report the out-of-sample SINR of the current policy.
        with torch.no_grad():
            sinr = eval_sinr_oos(w.detach(), spec, scenario["sig_theta"],
                                  scenario["interferers"], scenario["snr_db"],
                                  cal_err, device=args.device)
        history.append({
            "step": step,
            "loss": float(loss.item()),
            "sinr_db": float(sinr.item()),
        })
        if step % 200 == 0:
            recent = np.mean([h["sinr_db"] for h in history[-200:]])
            recent_loss = np.mean([h["loss"] for h in history[-200:]])
            print(f"  step {step:5d}  loss {recent_loss:.4f}  "
                  f"sinr {sinr.item():+.2f} dB  mean200 {recent:+.2f} dB")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": net.state_dict(),
        "config": {"n_elem": args.n_elem, "hidden": args.hidden},
        "history": history,
        "cal_max": args.cal_max,
    }, out)
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
