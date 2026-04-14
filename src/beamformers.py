"""Four beamformers with a uniform interface.

All of them map (R: sample covariance, a_nominal: assumed steering vector)
to a weight vector w ∈ C^n. The nominal steering vector is what a real
system "knows" — it does not see calibration errors. The caller computes
SINR using the *true* miscalibrated steering, which makes the comparison
fair between classical and learned methods.

  cbf      — Conventional beamformer (matched filter); ignores R
  mvdr     — Capon / minimum-variance distortionless response
  mvdr_dl  — MVDR with diagonal loading (robust classical baseline)
  NN       — MLP that takes R and a_nominal and emits w
"""
from __future__ import annotations
from typing import Callable
import torch
import torch.nn as nn


Beamformer = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def cbf(R: torch.Tensor, a_nominal: torch.Tensor) -> torch.Tensor:
    """Conventional beamformer: matched filter on the nominal steering."""
    n = a_nominal.shape[-1]
    return a_nominal / n


def mvdr(R: torch.Tensor, a_nominal: torch.Tensor,
         epsilon: float = 0.0) -> torch.Tensor:
    """Capon beamformer with optional diagonal loading."""
    n = a_nominal.shape[-1]
    R_reg = R + epsilon * torch.eye(n, dtype=R.dtype, device=R.device)
    Rinv_a = torch.linalg.solve(R_reg, a_nominal)
    denom = (a_nominal.conj() @ Rinv_a).real
    return Rinv_a / denom


def mvdr_dl(R: torch.Tensor, a_nominal: torch.Tensor,
            load_frac: float = 1.0) -> torch.Tensor:
    """MVDR with diagonal loading = load_frac * trace(R) / n * I.

    This is the classical "robust" MVDR. A larger load value pulls
    weights toward the conventional matched filter; smaller trusts
    R more.
    """
    n = a_nominal.shape[-1]
    load = load_frac * torch.trace(R).real / n
    return mvdr(R, a_nominal, epsilon=float(load.item()))


def _complex_to_real(x: torch.Tensor) -> torch.Tensor:
    """Stack real and imag parts along the last dim."""
    return torch.cat([x.real, x.imag], dim=-1)


def _real_to_complex(y: torch.Tensor) -> torch.Tensor:
    """Inverse of _complex_to_real."""
    n = y.shape[-1] // 2
    return y[..., :n] + 1j * y[..., n:]


class NNBeamformer(nn.Module):
    """Hybrid neural / classical beamformer.

    Rather than asking the MLP to learn the full `R⁻¹ · a` map from
    scratch — a hard nonlinear regression problem for a small network —
    the NN predicts two things:

      1. A diagonal-loading scalar ε (in log-space, so it's always positive)
      2. A small correction δa to the assumed steering vector a_nominal

    Given those, weights are computed by classical MVDR on
    `(R + εI, a_nominal + δa)`. This scheme:

      - has a tiny, well-conditioned output space
      - initializes essentially at MVDR-DL when the net is near zero,
        so training starts from a strong baseline rather than random
      - gives the NN a way to *compensate* for calibration errors by
        learning δa patterns that counteract typical per-element drift
      - still does a matrix solve per forward pass so the latency
        advantage over MVDR is modest; the win is robustness, not speed.

    The original "NN outputs full w directly" architecture is in
    `NNBeamformerDirect` below for completeness.
    """
    def __init__(self, n_elem: int, hidden: int = 128,
                 delta_scale: float = 0.1):
        super().__init__()
        self.n_elem = n_elem
        self.delta_scale = delta_scale
        n_upper = n_elem * (n_elem + 1) // 2
        in_dim = 2 * n_upper + 2 * n_elem
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1 + 2 * n_elem),
        )
        # Initialize output layer small so the NN starts near MVDR-DL.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _features(self, R: torch.Tensor, a_nominal: torch.Tensor) -> torch.Tensor:
        n = self.n_elem
        iu, ju = torch.triu_indices(n, n, device=R.device)
        flat = R[iu, ju]
        # Normalize by trace so feature magnitudes don't depend on signal power.
        scale = (torch.trace(R).real / n).clamp(min=1e-6)
        flat_r = _complex_to_real(flat / scale)
        a_r = _complex_to_real(a_nominal)
        return torch.cat([flat_r, a_r], dim=-1).to(torch.float32)

    def forward(self, R: torch.Tensor, a_nominal: torch.Tensor) -> torch.Tensor:
        n = self.n_elem
        x = self._features(R, a_nominal)
        out = self.net(x)

        # Bounded diagonal-loading scalar in [trace*1e-3, trace*10]
        trace_n = torch.trace(R).real / n
        log_eps = out[..., 0].to(torch.float64)
        eps = trace_n * torch.exp(log_eps.clamp(-6.0, 2.0)) * 0.1

        # Small learned correction to the assumed steering vector
        delta_real = out[..., 1:1 + n].to(torch.float64)
        delta_imag = out[..., 1 + n:].to(torch.float64)
        delta_a = self.delta_scale * (delta_real + 1j * delta_imag).to(torch.complex128)

        a_corrected = a_nominal + delta_a
        R_reg = R + eps * torch.eye(n, dtype=R.dtype, device=R.device)
        Rinv_a = torch.linalg.solve(R_reg, a_corrected)
        denom = (a_corrected.conj() @ Rinv_a).real
        return Rinv_a / (denom + 1e-12)


class NNBeamformerDirect(nn.Module):
    """Original free-form MLP that emits w directly. Kept for ablation."""
    def __init__(self, n_elem: int, hidden: int = 128):
        super().__init__()
        self.n_elem = n_elem
        n_upper = n_elem * (n_elem + 1) // 2
        in_dim = 2 * n_upper + 2 * n_elem
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * n_elem),
        )

    def forward(self, R: torch.Tensor, a_nominal: torch.Tensor) -> torch.Tensor:
        n = self.n_elem
        iu, ju = torch.triu_indices(n, n, device=R.device)
        flat = R[iu, ju]
        feats = torch.cat([_complex_to_real(flat), _complex_to_real(a_nominal)], -1)
        y = self.net(feats.to(torch.float32))
        w = _real_to_complex(y.to(torch.complex128))
        wa = (w.conj() * a_nominal).sum(dim=-1, keepdim=True)
        return w / (wa + 1e-12)
