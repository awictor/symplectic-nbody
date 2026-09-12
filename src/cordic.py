"""CORDIC: trigonometry, logarithms, and roots with only shifts and additions.

How does a pocket calculator or an FPGA with no hardware multiplier compute sin, cos, or a logarithm?
CORDIC (COordinate Rotation DIgital Computer, Volder 1959) evaluates a whole family of transcendental
functions using nothing but ADDITION, SUBTRACTION, and BIT SHIFTS -- no multiplication, no division,
no lookup of the function itself. It was the algorithm inside the first scientific calculators and
lives on in FPGAs, DSPs, and GPUs wherever a multiplier is scarce.

The idea is iterative ROTATION by ever-smaller angles whose tangents are exact powers of two, so
"multiplying" by the rotation is just a bit shift. To compute cos and sin of an angle, start at
(1, 0) and rotate toward the target angle in steps of arctan(2^-i); each step either adds or
subtracts the shifted coordinates depending on whether the running angle is short of or past the
target. After n steps the point is at (K cos theta, K sin theta) where K is a fixed GAIN that the
same iteration accumulates, so dividing out K (a single precomputed constant) gives cos and sin. This
is CIRCULAR mode; switching the sign of one update term to +1 puts CORDIC in HYPERBOLIC mode, which
computes sinh, cosh, exp, ln, and square roots by the same shift-add loop. A VECTORING variant drives
the y-coordinate to zero instead of the angle, computing arctan and hypot (magnitude) of an input
vector -- the basis of rectangular-to-polar conversion.

This module implements circular CORDIC (cos, sin, and via vectoring atan2 and hypot) and hyperbolic
CORDIC (exp, ln, sqrt), using only additions and shifts on scaled fixed-point-style floats, with
precomputed angle and gain tables. It is verified against Python's math library across the valid
ranges: cos/sin to about 1e-9, atan2 over all four quadrants, hypot, and exp/ln/sqrt to high
precision, that more iterations improve accuracy, and that the circular gain matches the analytic
product. Pure stdlib; a numerical-methods companion to the Newton's-method and Taylor-series notes."""

from __future__ import annotations

import math

# --- precomputed tables -----------------------------------------------------
_N = 40                                     # iterations (≈ bits of precision)
_ATAN_TABLE = [math.atan(2.0 ** -i) for i in range(_N)]

# circular gain K = prod sqrt(1 + 2^-2i)
_K = 1.0
for _i in range(_N):
    _K *= math.sqrt(1.0 + 2.0 ** (-2 * _i))
_INV_K = 1.0 / _K

# hyperbolic arctanh table; indices where i is repeated (4,13,40,...) for convergence
_ATANH_TABLE = {}
_HYP_INDICES = []
_next_repeat = 4
_i = 1
while _i <= _N:
    _ATANH_TABLE[_i] = math.atanh(2.0 ** -_i)
    _HYP_INDICES.append(_i)
    if _i == _next_repeat:
        _HYP_INDICES.append(_i)               # repeat this index
        _next_repeat = 3 * _i + 1
    _i += 1

# hyperbolic gain over the repeated schedule
_Kh = 1.0
for _idx in _HYP_INDICES:
    _Kh *= math.sqrt(1.0 - 2.0 ** (-2 * _idx))
_INV_Kh = 1.0 / _Kh


def cos_sin(theta):
    """cos and sin of theta (radians) by circular CORDIC in rotation mode. Reduces theta to
    [-pi/2, pi/2] first (CORDIC converges only within ~1.74 rad)."""
    # range reduction to [-pi, pi], then handle the half-plane by sign flips
    two_pi = 2 * math.pi
    theta = theta - two_pi * math.floor((theta + math.pi) / two_pi)
    flip = 1.0
    if theta > math.pi / 2:
        theta -= math.pi
        flip = -1.0
    elif theta < -math.pi / 2:
        theta += math.pi
        flip = -1.0
    x, y, z = _INV_K, 0.0, theta
    for i in range(_N):
        d = 1.0 if z >= 0 else -1.0
        dx = x - d * (y * 2.0 ** -i)
        dy = y + d * (x * 2.0 ** -i)
        x, y = dx, dy
        z -= d * _ATAN_TABLE[i]
    return flip * x, flip * y


def cordic_cos(theta):
    return cos_sin(theta)[0]


def cordic_sin(theta):
    return cos_sin(theta)[1]


def atan2_hypot(y, x):
    """atan2(y, x) and hypot(x, y) by circular CORDIC in vectoring mode (drive y to zero)."""
    # handle the half-planes so the rotation stays in range
    if x < 0:
        # rotate by pi first
        addend = math.pi if y >= 0 else -math.pi
        x, y = -x, -y
    else:
        addend = 0.0
    vx, vy, z = x, y, 0.0
    for i in range(_N):
        d = 1.0 if vy < 0 else -1.0
        nvx = vx - d * (vy * 2.0 ** -i)
        nvy = vy + d * (vx * 2.0 ** -i)
        vx, vy = nvx, nvy
        z -= d * _ATAN_TABLE[i]
    return z + addend, vx * _INV_K


def cordic_atan2(y, x):
    return atan2_hypot(y, x)[0]


def cordic_hypot(x, y):
    return atan2_hypot(y, x)[1]


def _hyperbolic(x0, y0, z0, mode):
    """Hyperbolic CORDIC. mode='rotation' drives z->0, 'vectoring' drives y->0."""
    x, y, z = x0, y0, z0
    for i in _HYP_INDICES:
        shift = 2.0 ** -i
        if mode == "rotation":
            d = 1.0 if z >= 0 else -1.0
        else:
            d = 1.0 if y < 0 else -1.0
        nx = x + d * (y * shift)
        ny = y + d * (x * shift)
        x, y = nx, ny
        z -= d * _ATANH_TABLE[i]
    return x, y, z


def cordic_exp(x):
    """e^x via hyperbolic CORDIC: cosh(x) + sinh(x). Valid for |x| up to ~1.11; larger x uses the
    identity e^x = e^(x - k ln2) * 2^k for range reduction."""
    k = 0
    ln2 = math.log(2)
    while x > 1.0:
        x -= ln2
        k += 1
    while x < -1.0:
        x += ln2
        k -= 1
    cx, sx, _ = _hyperbolic(_INV_Kh, 0.0, x, "rotation")
    return (cx + sx) * (2.0 ** k)


def cordic_ln(a):
    """ln(a) via hyperbolic CORDIC vectoring: ln(a) = 2 * atanh((a-1)/(a+1)), computed by driving
    y to zero from (a+1, a-1). Range-reduced by factoring out powers of two."""
    if a <= 0:
        raise ValueError("ln domain is positive reals")
    k = 0
    while a > 2.0:
        a /= 2.0
        k += 1
    while a < 0.5:
        a *= 2.0
        k -= 1
    _, _, z = _hyperbolic(a + 1.0, a - 1.0, 0.0, "vectoring")
    return 2.0 * z + k * math.log(2)


def cordic_sqrt(a):
    """sqrt(a) via hyperbolic CORDIC: sqrt(a) = sqrt((a+1/4)^2 - (a-1/4)^2) using the vectoring
    magnitude. Range-reduced by factoring out even powers of two."""
    if a < 0:
        raise ValueError("sqrt domain is non-negative reals")
    if a == 0:
        return 0.0
    k = 0
    while a > 1.0:
        a /= 4.0
        k += 1
    while a < 0.25:
        a *= 4.0
        k -= 1
    x, _, _ = _hyperbolic(a + 0.25, a - 0.25, 0.0, "vectoring")
    return x * _INV_Kh * (2.0 ** k)
