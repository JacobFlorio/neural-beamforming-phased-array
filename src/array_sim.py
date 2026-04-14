"""Uniform linear array (ULA) simulator with optional calibration errors.

All math is in torch (complex128) so the full pipeline — array data →
sample covariance → beamformer weights → SINR — is differentiable end
to end. This lets us train a neural beamformer by backpropagation
through the SINR objective.

Two things that are important and easy to get wrong:

1. **Calibration error.** Every real array has per-element gain and
   phase mismatches (LNA variation, cable length, temperature drift).
   The *assumed* steering vector is the nominal `[1, e^{-jπ sin θ}, ...]`;
   the *true* steering vector is that multiplied elementwise by a
   per-element complex error `g_m e^{jφ_m}`. Beamformers built on the
   assumed model can null the signal of interest by mistake when
   calibration is wrong. Injecting this error is the whole point of
   the "robustness" experiment.

2. **In-sample vs out-of-sample SINR.** If you compute SINR using the
   same snapshots that built the sample covariance R, the number you
   get is the *training* SINR — optimistic by construction because
   MVDR can fit the noise realization. The honest metric evaluates on
   a fresh draw of snapshots. This module's `eval_sinr_oos` does that.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class ArraySpec:
    n_elem: int = 16
    d_lambda: float = 0.5


def steering_vector(spec: ArraySpec, theta_rad: torch.Tensor) -> torch.Tensor:
    """Nominal (calibration-error-free) steering vector for a ULA.

    theta_rad: scalar or [B] batch of angles.
    returns: [n_elem] or [B, n_elem] complex128 tensor.
    """
    m = torch.arange(spec.n_elem, dtype=torch.float64, device=theta_rad.device)
    if theta_rad.dim() == 0:
        phase = -2 * torch.pi * spec.d_lambda * m * torch.sin(theta_rad)
        return torch.exp(1j * phase.to(torch.complex128))
    # batched
    phase = -2 * torch.pi * spec.d_lambda * m[None, :] * torch.sin(theta_rad)[:, None]
    return torch.exp(1j * phase.to(torch.complex128))


def calibration_error(spec: ArraySpec, magnitude: float,
                       generator: torch.Generator | None = None,
                       device: str = "cpu") -> torch.Tensor:
    """Per-element complex gain/phase error.

    magnitude is a single scalar that controls both gain std and phase
    std (in radians). magnitude=0 is no error; magnitude=0.1 is roughly
    10% gain error and 0.1 rad phase error.
    returns: [n_elem] complex128 tensor with |...| ~ 1.
    """
    gen = generator
    gains = 1.0 + magnitude * torch.randn(spec.n_elem, generator=gen, device=device, dtype=torch.float64)
    phases = magnitude * torch.randn(spec.n_elem, generator=gen, device=device, dtype=torch.float64)
    return (gains * torch.exp(1j * phases.to(torch.complex128)))


def true_steering(spec: ArraySpec, theta_rad: torch.Tensor,
                   cal_err: torch.Tensor) -> torch.Tensor:
    """Steering vector with per-element calibration error applied."""
    nominal = steering_vector(spec, theta_rad)
    if nominal.dim() == 1:
        return nominal * cal_err
    return nominal * cal_err[None, :]


def simulate_snapshots(spec: ArraySpec, n_snap: int,
                        sig_theta: float,
                        interferers: list[tuple[float, float]],
                        snr_db: float,
                        cal_err: torch.Tensor | None = None,
                        generator: torch.Generator | None = None,
                        device: str = "cpu"):
    """Generate a batch of snapshots with a signal, K interferers, and noise.

    Returns:
        X:       [n_elem, n_snap] complex snapshots
        R:       [n_elem, n_snap] -> [n_elem, n_elem] sample covariance
        a_true:  [n_elem] true steering vector for the signal (cal-error applied)
        a_nominal: [n_elem] nominal (what the beamformer "knows")
    """
    gen = generator
    n = spec.n_elem
    if cal_err is None:
        cal_err = torch.ones(n, dtype=torch.complex128, device=device)

    theta_s = torch.tensor(sig_theta, dtype=torch.float64, device=device)
    a_true = true_steering(spec, theta_s, cal_err)
    a_nominal = steering_vector(spec, theta_s)

    # Gaussian signal
    s = (torch.randn(n_snap, generator=gen, device=device, dtype=torch.float64) +
         1j * torch.randn(n_snap, generator=gen, device=device, dtype=torch.float64)) / (2 ** 0.5)
    X = a_true[:, None] * s[None, :]

    for theta, inr_db in interferers:
        theta_t = torch.tensor(theta, dtype=torch.float64, device=device)
        a_i = true_steering(spec, theta_t, cal_err)
        gain = 10 ** (inr_db / 20)
        i = (torch.randn(n_snap, generator=gen, device=device, dtype=torch.float64) +
             1j * torch.randn(n_snap, generator=gen, device=device, dtype=torch.float64)) / (2 ** 0.5)
        X = X + a_i[:, None] * (gain * i)[None, :]

    sigma_n = 10 ** (-snr_db / 20)
    noise = (torch.randn(n, n_snap, generator=gen, device=device, dtype=torch.float64) +
             1j * torch.randn(n, n_snap, generator=gen, device=device, dtype=torch.float64)) / (2 ** 0.5)
    X = X + sigma_n * noise

    R = X @ X.conj().T / n_snap
    return X, R, a_true, a_nominal


def sinr_from_weights(w: torch.Tensor, a_true: torch.Tensor,
                       R_in: torch.Tensor, sig_power: float = 1.0) -> torch.Tensor:
    """Analytical SINR given a weight vector and the true (noise+interf) covariance.

    R_in is the interference + noise covariance *without* the signal
    contribution. This is the cleanest out-of-sample SINR definition:
    signal power = |w^H a_true|^2, interference+noise power = w^H R_in w.
    """
    wa = (w.conj() * a_true).sum(dim=-1)
    signal = sig_power * (wa.conj() * wa).real
    wRw = (w.conj() @ R_in @ w).real
    eps = 1e-12
    return 10.0 * torch.log10((signal + eps) / (wRw + eps))


def r_interference_plus_noise(spec: ArraySpec,
                               interferers: list[tuple[float, float]],
                               snr_db: float,
                               cal_err: torch.Tensor,
                               device: str = "cpu") -> torch.Tensor:
    """Analytical R_i+n = Σ P_i a_i a_i^H + σ² I.

    Uses the TRUE (miscalibrated) steering vectors for the interferers.
    This is what you'd compute if you had infinite snapshots.
    """
    n = spec.n_elem
    R = torch.zeros(n, n, dtype=torch.complex128, device=device)
    for theta, inr_db in interferers:
        theta_t = torch.tensor(theta, dtype=torch.float64, device=device)
        a = true_steering(spec, theta_t, cal_err)
        P = 10 ** (inr_db / 10)
        R = R + P * torch.outer(a, a.conj())
    sigma2 = 10 ** (-snr_db / 10)
    R = R + sigma2 * torch.eye(n, dtype=torch.complex128, device=device)
    return R


def eval_sinr_oos(w: torch.Tensor, spec: ArraySpec,
                   sig_theta: float,
                   interferers: list[tuple[float, float]],
                   snr_db: float,
                   cal_err: torch.Tensor,
                   device: str = "cpu") -> torch.Tensor:
    """Out-of-sample SINR: analytical formula using the true covariance.

    This is the honest metric — no leakage of the same snapshots that
    built the sample-R back into the SINR number.
    """
    a_true = true_steering(spec, torch.tensor(sig_theta, dtype=torch.float64, device=device), cal_err)
    R_in = r_interference_plus_noise(spec, interferers, snr_db, cal_err, device)
    return sinr_from_weights(w, a_true, R_in)
