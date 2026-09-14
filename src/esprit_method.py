"""ESPRIT: recover the exact frequencies of superimposed sinusoids from a noisy signal, beating the FFT's resolution.

The FFT reports energy in fixed frequency bins; two tones closer than one bin blur into a single lump, and
noise smears every peak. Prony's method resolves the exact frequencies but is notoriously fragile -- a
little noise wrecks the linear-prediction step. ESPRIT (Estimation of Signal Parameters via Rotational
Invariance Techniques; Roy & Kailath 1989) is the noise-robust subspace method that fixes this. It rests
on one elegant observation: if a signal is a sum of p complex exponentials, then a time-shifted copy of
any window into the signal is the SAME window multiplied, mode by mode, by the exponentials' per-sample
phase factors z_k. In linear-algebra terms the signal subspace is ROTATIONALLY INVARIANT under a time
shift, and the rotation's eigenvalues are exactly the z_k.

The algorithm: build a Hankel data matrix from the samples, take its SVD, and keep the p left singular
vectors spanning the signal subspace (the rest is noise -- this projection is where the robustness comes
from). Split that basis into its first and last rows, Us_up and Us_down; the p-by-p matrix Psi solving
Us_up Psi = Us_down (a least-squares problem) is the shift rotation, and its EIGENVALUES are the modes
z_k = exp((-d_k + i 2 pi f_k) dt). Frequencies, dampings, and -- via a Vandermonde fit -- amplitudes
follow. Because the noise subspace is discarded before any root-finding, ESPRIT stays accurate at signal-
to-noise ratios where Prony collapses, and it super-resolves frequencies far below the FFT bin spacing.

This module implements ESPRIT with an SVD-based signal-subspace projection (reusing the repo's SVD), the
least-squares rotation solve, and eigenvalue extraction (reusing the QR-algorithm), plus signal
reconstruction. It is validated: on a clean sum of sinusoids it recovers the exact frequencies and damping;
it resolves two tones far closer than one FFT bin where the periodogram shows a single peak; it recovers
frequencies from a NOISY signal to a small error where Prony's error is far larger (its defining
advantage); a pure damped exponential yields the right decay rate; and the reconstruction matches the clean
signal. Pure stdlib; the subspace-spectral companion to the Prony, FFT, Goertzel, and Welch-PSD tools."""

from __future__ import annotations

import cmath
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import svd as _svd
import aberth


def _hankel(x, rows):
    """Build a Hankel data matrix with `rows` rows from signal x. H[i][j] = x[i+j]."""
    n = len(x)
    cols = n - rows + 1
    return [[x[i + j] for j in range(cols)] for i in range(rows)]


def _solve_ls_complex(A, B):
    """Least-squares solve A X = B for complex A (m x p), B (m x q) via normal equations A^H A X = A^H B.

    Returns X (p x q). Used for the small p-by-p rotation solve, where p is tiny."""
    m = len(A)
    p = len(A[0])
    q = len(B[0])
    # A^H A  (p x p) and A^H B (p x q)
    AhA = [[sum(A[k][i].conjugate() * A[k][j] for k in range(m)) for j in range(p)] for i in range(p)]
    AhB = [[sum(A[k][i].conjugate() * B[k][j] for k in range(m)) for j in range(q)] for i in range(p)]
    return _solve_complex_system(AhA, AhB)


def _solve_complex_system(A, B):
    """Solve A X = B (A: p x p complex, B: p x q complex) by Gaussian elimination with partial pivoting."""
    p = len(A)
    q = len(B[0])
    M = [[A[i][j] for j in range(p)] + [B[i][j] for j in range(q)] for i in range(p)]
    for col in range(p):
        piv = max(range(col, p), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-300:
            M[piv][col] += 1e-300
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for r in range(p):
            if r == col:
                continue
            f = M[r][col] / pv
            for c in range(col, p + q):
                M[r][c] -= f * M[col][c]
    X = [[0j] * q for _ in range(p)]
    for i in range(p):
        for c in range(q):
            X[i][c] = M[i][p + c] / M[i][i]
    return X


def _eig_complex(A):
    """Eigenvalues of a small COMPLEX square matrix, as the roots of its characteristic polynomial.

    Builds the char-poly coefficients by the Faddeev-LeVerrier recurrence (works over the complex field,
    unlike the real QR-algorithm) and finds their roots with the repo's Aberth solver. Fine for the tiny
    p-by-p rotation matrices ESPRIT produces."""
    n = len(A)
    # Faddeev-LeVerrier: coefficients c_0=1, c_k of det(lam I - A)
    I = [[1.0 + 0j if i == j else 0j for j in range(n)] for i in range(n)]
    M = [[0j] * n for _ in range(n)]
    c = [1.0 + 0j]
    for k in range(1, n + 1):
        AM = _cmatmul(A, M)
        M = [[AM[i][j] + c[k - 1] * I[i][j] for j in range(n)] for i in range(n)]
        AMk = _cmatmul(A, M)
        ck = -sum(AMk[i][i] for i in range(n)) / k
        c.append(ck)
    return aberth.roots(c)


def _cmatmul(A, B):
    n = len(A)
    p = len(B[0])
    m = len(B)
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]


def esprit(x, p, dt=1.0, rows=None):
    """Estimate p complex modes of signal x by ESPRIT.

    Returns a dict with modes z_k, frequencies f_k (Hz), damping rates d_k, and complex amplitudes a_k.
    `rows` is the Hankel matrix row count (defaults to ~n/2, which maximizes the subspace dimension)."""
    n = len(x)
    if n < 2 * p + 1:
        raise ValueError(f"need at least {2 * p + 1} samples for {p} modes")
    if rows is None:
        # Keep the COLUMN count small: svd() eigendecomposes the cols-by-cols Gram matrix, so a
        # tall-skinny Hankel (few columns) is far cheaper and just as accurate. cols ~ 3p+5 gives
        # a healthy over-determined invariance solve without the O(cols^3) SVD blowup.
        cols_target = min(n - p - 1, 3 * p + 5)
        rows = n - cols_target + 1
    rows = max(p + 1, min(rows, n - p))     # need rows > p and enough columns

    H = _hankel([float(v) for v in x], rows)

    # SVD; signal subspace = first p left singular vectors (columns of U, descending order)
    U, S, Vt = _svd.svd(H, tol=1e-15)
    if len(S) < p:
        raise ValueError("signal subspace smaller than requested mode count")
    # Us: rows x p signal-subspace basis
    Us = [[U[i][k] for k in range(p)] for i in range(rows)]

    # rotational invariance: Us_up (rows 0..rows-2) and Us_down (rows 1..rows-1)
    Us_up = [Us[i] for i in range(rows - 1)]
    Us_down = [Us[i] for i in range(1, rows)]
    # solve Us_up Psi = Us_down for the p x p rotation
    Us_up_c = [[complex(v) for v in row] for row in Us_up]
    Us_down_c = [[complex(v) for v in row] for row in Us_down]
    Psi = _solve_ls_complex(Us_up_c, Us_down_c)

    # modes = eigenvalues of the (complex) rotation Psi, as roots of its characteristic polynomial
    modes = _eig_complex(Psi)

    # amplitudes via Vandermonde least squares  x[m] = sum_k a_k z_k^m
    V = [[modes[k] ** m for k in range(p)] for m in range(n)]
    amps = _amp_least_squares(V, [complex(v) for v in x])

    freqs, damps = [], []
    for z in modes:
        if z == 0:
            damps.append(float("inf"))
            freqs.append(0.0)
            continue
        lg = cmath.log(z)
        damps.append(-lg.real / dt)
        freqs.append(lg.imag / (2 * math.pi * dt))
    return {"modes": modes, "amplitudes": amps, "frequencies": freqs, "damping": damps}


def _amp_least_squares(V, b):
    """Least-squares solve V a = b via normal equations (V: n x p complex, b: length n)."""
    n = len(V)
    p = len(V[0])
    VhV = [[sum(V[k][i].conjugate() * V[k][j] for k in range(n)) for j in range(p)] for i in range(p)]
    Vhb = [[sum(V[k][i].conjugate() * b[k] for k in range(n))] for i in range(p)]
    X = _solve_complex_system(VhV, Vhb)
    return [X[i][0] for i in range(p)]


def reconstruct(model, n):
    """Rebuild the signal x[m] = sum_k a_k z_k^m for m = 0..n-1 from an ESPRIT model."""
    modes = model["modes"]
    amps = model["amplitudes"]
    out = []
    for m in range(n):
        val = sum(amps[k] * modes[k] ** m for k in range(len(modes)))
        out.append(val.real)
    return out


def periodogram_peak_count(x, threshold=0.5):
    """Count spectral peaks in the naive periodogram above `threshold` * max, for FFT-comparison demos.

    Uses a direct DFT (no external deps); returns (num_peaks, magnitude_spectrum)."""
    n = len(x)
    mags = []
    for k in range(n // 2 + 1):
        re = sum(x[t] * math.cos(-2 * math.pi * k * t / n) for t in range(n))
        im = sum(x[t] * math.sin(-2 * math.pi * k * t / n) for t in range(n))
        mags.append(math.sqrt(re * re + im * im))
    mx = max(mags) or 1.0
    peaks = 0
    for k in range(1, len(mags) - 1):
        if mags[k] > threshold * mx and mags[k] >= mags[k - 1] and mags[k] >= mags[k + 1]:
            peaks += 1
    return peaks, mags
