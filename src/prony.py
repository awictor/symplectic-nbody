"""Prony's method: decomposing a signal into a sum of damped complex exponentials.

The Fourier transform describes a signal by its frequency content on a fixed grid of bins, but a
signal that is genuinely a sum of a FEW damped sinusoids -- a ringing bell, a decaying circuit
transient, two close spectral lines -- is described far more compactly by their exact frequencies,
damping rates, amplitudes, and phases. PRONY'S METHOD (1795, older than Fourier's paper) fits exactly
that model,

    x[n] = sum_{k=1}^{p} a_k * z_k^n,   z_k = exp((-d_k + i*2*pi*f_k) * dt),

to a sampled signal, recovering the p complex modes z_k (hence each frequency f_k and damping d_k) and
their complex amplitudes a_k (hence magnitude and phase). It is the parametric, super-resolution
cousin of the FFT: it can separate two frequencies closer than the FFT's bin spacing, because it does
not bin at all -- it solves for the poles directly.

The algorithm is a neat three-step linear procedure. First, the samples satisfy a linear recurrence
whose characteristic polynomial has the z_k as roots; its coefficients come from a
least-squares (or exact, for 2p samples) linear system built from the data (a Hankel system). Second,
find the roots of that polynomial -- the modes z_k. Third, the amplitudes a_k solve a Vandermonde
least-squares fit of the data to the recovered modes. This module implements all three, using
companion-matrix eigenvalues for the roots and a normal-equations solve for the amplitudes, and
reconstructs the signal from the fitted model.

Validated: on a signal built as a known sum of damped sinusoids, Prony recovers the exact frequencies
and damping rates and reconstructs the samples to machine precision; a pure undamped tone gives zero
damping and the right frequency; two frequencies closer than one FFT bin are resolved where the FFT
peak is a single blob; and a single real exponential recovers its decay rate. Pure stdlib; the
parametric-spectral companion to the FFT, Goertzel, and Levinson-Durbin tools."""

from __future__ import annotations

import cmath
import math

from durand_kerner import roots as _dk_roots


def _solve_complex(A, b):
    """Gaussian elimination with partial pivoting for a complex dense system."""
    n = len(b)
    M = [[complex(A[i][j]) for j in range(n)] + [complex(b[i])] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        p = M[col][col]
        if abs(p) < 1e-15:
            continue
        for r in range(col + 1, n):
            f = M[r][col] / p
            for c in range(col, n + 1):
                M[r][c] -= f * M[col][c]
    x = [0j] * n
    for i in range(n - 1, -1, -1):
        if abs(M[i][i]) < 1e-15:
            x[i] = 0j
            continue
        s = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / M[i][i]
    return x


def _least_squares_complex(A, b):
    """Solve the (possibly overdetermined) complex system A x = b by normal equations A^H A x = A^H b."""
    m = len(A)
    n = len(A[0])
    # A^H A and A^H b
    AhA = [[sum(A[k][i].conjugate() * A[k][j] for k in range(m)) for j in range(n)] for i in range(n)]
    Ahb = [sum(A[k][i].conjugate() * b[k] for k in range(m)) for i in range(n)]
    return _solve_complex(AhA, Ahb)


def prony(x, p, dt=1.0):
    """Fit x[n] = sum_k a_k z_k^n with p modes. Returns a dict with modes z_k, complex amplitudes
    a_k, frequencies f_k (Hz), and damping rates d_k."""
    n = len(x)
    if n < 2 * p:
        raise ValueError(f"need at least {2 * p} samples for {p} modes")
    # Step 1: linear-predictor coefficients from the Hankel system.
    # x[m] = -sum_{j=1}^p a_j x[m-j], for m = p..n-1  (a monic polynomial with roots z_k)
    A = []
    b = []
    for m in range(p, n):
        A.append([x[m - j] for j in range(1, p + 1)])
        b.append(-x[m])
    # least-squares for the predictor coefficients (real -> promote to complex solve)
    coef = _least_squares_complex([[complex(v) for v in row] for row in A], [complex(v) for v in b])
    # characteristic polynomial z^p + coef[0] z^{p-1} + ... + coef[p-1] = 0, highest degree first
    poly_hi_first = [1.0] + [coef[j] for j in range(p)]
    modes = _dk_roots(poly_hi_first)
    # Step 3: amplitudes via Vandermonde least squares  x[n] = sum a_k z_k^n
    V = [[modes[k] ** nn for k in range(len(modes))] for nn in range(n)]
    amps = _least_squares_complex(V, [complex(v) for v in x])
    freqs = []
    damps = []
    for z in modes:
        # z = exp((-d + i 2 pi f) dt)
        d = -cmath.log(z).real / dt
        f = cmath.log(z).imag / (2 * math.pi * dt)
        damps.append(d)
        freqs.append(f)
    return {"modes": modes, "amplitudes": amps, "frequencies": freqs, "damping": damps}


def reconstruct(model, n):
    """Reconstruct n samples from a Prony model."""
    modes = model["modes"]
    amps = model["amplitudes"]
    out = []
    for nn in range(n):
        s = sum(amps[k] * modes[k] ** nn for k in range(len(modes)))
        out.append(s.real)
    return out
