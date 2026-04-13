"""Uniform linear array (ULA) signal generator with interferers.

Produces sample covariance matrices R = (1/N) X X^H that a beamformer
can consume. Includes an analytic MVDR (Capon) reference for baseline.
"""
from __future__ import annotations
import numpy as np


def steering_vector(n_elem: int, theta_rad: float, d_lambda: float = 0.5) -> np.ndarray:
    m = np.arange(n_elem)
    return np.exp(-1j * 2 * np.pi * d_lambda * m * np.sin(theta_rad))


def simulate(n_elem: int = 16, n_snap: int = 256, sig_theta: float = 0.0,
             interferers: list[tuple[float, float]] | None = None,
             snr_db: float = 10.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    interferers = interferers or []
    s = (rng.standard_normal(n_snap) + 1j * rng.standard_normal(n_snap)) / np.sqrt(2)
    a_s = steering_vector(n_elem, sig_theta)
    X = a_s[:, None] * s[None, :]
    for theta, inr_db in interferers:
        i = (rng.standard_normal(n_snap) + 1j * rng.standard_normal(n_snap)) / np.sqrt(2)
        i *= 10 ** (inr_db / 20)
        X += steering_vector(n_elem, theta)[:, None] * i[None, :]
    sigma_n = 10 ** (-snr_db / 20)
    noise = (rng.standard_normal(X.shape) + 1j * rng.standard_normal(X.shape)) / np.sqrt(2)
    X += sigma_n * noise
    R = (X @ X.conj().T) / n_snap
    return X, R, a_s


def mvdr_weights(R: np.ndarray, a: np.ndarray) -> np.ndarray:
    R_inv = np.linalg.inv(R + 1e-6 * np.eye(R.shape[0]))
    num = R_inv @ a
    return num / (a.conj() @ num)


def sinr(w: np.ndarray, R: np.ndarray, a_s: np.ndarray, sig_power: float = 1.0) -> float:
    wa = w.conj() @ a_s
    signal = sig_power * np.abs(wa) ** 2
    total = float((w.conj() @ R @ w).real)
    noise_interf = max(total - signal, 1e-12)
    return 10 * np.log10(signal / noise_interf)
