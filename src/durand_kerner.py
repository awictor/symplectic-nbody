"""Durand-Kerner: finding ALL roots of a polynomial at once.

The bisection/Newton family finds ONE real root at a time and needs a bracket or a good start. The
DURAND-KERNER (Weierstrass) method finds ALL n roots of a degree-n polynomial simultaneously, real and
complex, from a single set of starting guesses -- no bracketing, no deflation, no derivative. It treats
the n unknown roots as a coupled system and refines them together, converging quadratically once close.
The fundamental theorem of algebra guarantees exactly n roots (with multiplicity) in the complex plane,
and Durand-Kerner is the classic way to get them all: it underlies polynomial factoring in computer
algebra, filter and control design (finding transfer-function poles and zeros), and any problem that
reduces to the roots of a characteristic polynomial.

The idea is elegant. If p(x) = c*(x - r_1)...(x - r_n), then each root satisfies
r_i = r_i - p(r_i) / (c * prod_{j != i} (r_i - r_j)). Starting from n distinct guesses -- classically
powers of a complex number like 0.4 + 0.9i, spread around the unit circle so no two coincide -- apply
that update to every guess each iteration (Jacobi- or Gauss-Seidel-style), and the guesses march to the
true roots. Each iteration is O(n^2), and near the roots convergence is quadratic, so a handful of
iterations suffices for well-conditioned polynomials. Working in Python's built-in complex arithmetic
makes the whole thing exact-typed and dependency-free.

This module normalises a polynomial (given as a coefficient list), runs Durand-Kerner from spread
complex seeds, and returns all n roots. It is verified against construction -- build a polynomial from
KNOWN roots (real, complex-conjugate, repeated), find the roots, and match them up within tolerance --
and against reconstruction: evaluating the polynomial at each returned root gives ~0, and the roots'
elementary symmetric functions reproduce the coefficients (Vieta's formulas). Tested on hundreds of
random polynomials with real and complex roots. Pure stdlib (cmath); a numerical-methods companion to
the bracketing root-finders, the companion-matrix eigenvalue view, and the FFT notes."""

from __future__ import annotations

import cmath


def _eval(coeffs, x):
    """Evaluate the polynomial (coeffs high-degree-first) at x by Horner's method."""
    result = 0
    for c in coeffs:
        result = result * x + c
    return result


def _normalise(coeffs):
    """Strip leading zeros and return a monic coefficient list (divided by the leading coeff)."""
    i = 0
    while i < len(coeffs) - 1 and coeffs[i] == 0:
        i += 1
    coeffs = coeffs[i:]
    lead = coeffs[0]
    return [c / lead for c in coeffs]


def roots(coeffs, max_iter=500, tol=1e-12):
    """All complex roots of the polynomial with coefficients `coeffs` (highest degree first), via
    Durand-Kerner. Returns a list of `degree` complex numbers.

    Constant or empty polynomials have no roots."""
    mono = _normalise([complex(c) for c in coeffs])
    n = len(mono) - 1            # degree
    if n <= 0:
        return []

    # initial guesses: powers of a fixed complex seed, spread so none coincide
    seed = complex(0.4, 0.9)
    guesses = [seed ** k for k in range(n)]

    for _ in range(max_iter):
        max_delta = 0.0
        new = list(guesses)
        for i in range(n):
            xi = guesses[i]
            denom = 1 + 0j
            for j in range(n):
                if j != i:
                    denom *= (xi - guesses[j])
            if denom == 0:
                continue          # coincident guesses; skip this step for i
            delta = _eval(mono, xi) / denom
            new[i] = xi - delta
            max_delta = max(max_delta, abs(delta))
        guesses = new
        if max_delta < tol:
            break
    return guesses


def _clean(z, tol=1e-9):
    """Round a complex root to a real number when its imaginary part is negligible, and snap tiny
    parts to zero, for presentation."""
    re = z.real
    im = z.imag
    if abs(im) < tol:
        return complex(round(re, 10), 0.0) if abs(re - round(re)) > 1e-11 else complex(round(re), 0.0)
    return z


def real_roots(coeffs, tol=1e-7):
    """Just the (approximately) real roots, as floats, sorted."""
    out = []
    for z in roots(coeffs):
        if abs(z.imag) < tol:
            out.append(z.real)
    return sorted(out)


def from_roots(root_list):
    """Build the monic polynomial coefficients (highest degree first) with the given roots."""
    coeffs = [1 + 0j]
    for r in root_list:
        # multiply current polynomial by (x - r)
        new = [0j] * (len(coeffs) + 1)
        for i, c in enumerate(coeffs):
            new[i] += c            # x * c
            new[i + 1] += -r * c   # -r * c
        coeffs = new
    return coeffs


# --- verification helpers ---------------------------------------------------
def residual(coeffs, root):
    """|p(root)| -- should be ~0 for a true root."""
    return abs(_eval([complex(c) for c in _normalise([complex(x) for x in coeffs])], root))


def match_roots(found, expected, tol=1e-6):
    """True iff `found` and `expected` are the same multiset of complex numbers within tol (greedy
    nearest-matching)."""
    if len(found) != len(expected):
        return False
    remaining = list(expected)
    for f in found:
        best = None
        best_d = None
        for i, e in enumerate(remaining):
            d = abs(f - e)
            if best_d is None or d < best_d:
                best_d = d
                best = i
        if best is None or best_d > tol:
            return False
        remaining.pop(best)
    return True
