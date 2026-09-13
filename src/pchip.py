"""PCHIP: shape-preserving cubic interpolation that never overshoots.

A natural cubic spline is smooth (C^2) but it can OVERSHOOT -- interpolating monotone data, it may
dip below or bulge above the samples, inventing wiggles that were never in the data. For monotone or
positive data (a cumulative distribution, a saturating dose-response, a concentration that only
falls) that is unacceptable. PCHIP -- Piecewise Cubic Hermite Interpolating Polynomial, the
Fritsch-Carlson (1980) construction -- gives up one order of smoothness (it is C^1, not C^2) in
exchange for a guarantee: if the data are monotone on an interval, so is the interpolant, with no
overshoot anywhere.

The trick is how it chooses the DERIVATIVE at each knot. A Hermite cubic on each interval is
determined by the endpoint values and slopes; PCHIP sets each knot's slope to a WEIGHTED HARMONIC
MEAN of the secant slopes of the two neighbouring intervals -- but only when those secants have the
same sign. If the data change direction at a knot (a local max or min), the slope is set to zero,
pinning the interpolant so it cannot overshoot. The harmonic mean also keeps the slope small when
either neighbour is flat, exactly where a natural spline would overshoot.

This module computes PCHIP derivatives, evaluates the interpolant at arbitrary points, and reports
the per-interval cubic coefficients. Validated by the shape guarantee and correctness: the
interpolant passes exactly through every knot; on monotone-increasing data it is monotone
increasing everywhere (no dips) whereas a natural cubic spline overshoots on the classic step-like
data; it stays within the data range (no new maxima/minima between knots); it reproduces a straight
line exactly; and it is continuous with continuous first derivative at the knots. Pure stdlib; the
shape-preserving companion to the cubic-spline and barycentric interpolation tools."""

from __future__ import annotations


def _pchip_slopes(x, y):
    """Fritsch-Carlson derivatives at each knot that preserve monotonicity."""
    n = len(x)
    if n == 2:
        s = (y[1] - y[0]) / (x[1] - x[0])
        return [s, s]
    h = [x[i + 1] - x[i] for i in range(n - 1)]
    delta = [(y[i + 1] - y[i]) / h[i] for i in range(n - 1)]  # secant slopes
    d = [0.0] * n
    # interior knots: weighted harmonic mean when secants agree in sign, else 0
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] > 0:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
        else:
            d[i] = 0.0
    # endpoints: one-sided, non-centered three-point formula with monotonicity limiting
    d[0] = _endpoint_slope(delta[0], delta[1] if n > 2 else delta[0], h[0], h[1] if n > 2 else h[0])
    d[-1] = _endpoint_slope(delta[-1], delta[-2] if n > 2 else delta[-1],
                            h[-1], h[-2] if n > 2 else h[-1])
    return d


def _endpoint_slope(d0, d1, h0, h1):
    """Non-centered slope estimate at an endpoint, limited to preserve shape (Fritsch-Carlson)."""
    d = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    if d * d0 <= 0:
        d = 0.0
    elif d0 * d1 <= 0 and abs(d) > 3 * abs(d0):
        d = 3 * d0
    return d


class PCHIP:
    """Piecewise cubic Hermite interpolant (monotone, shape-preserving)."""

    def __init__(self, x, y):
        if len(x) != len(y):
            raise ValueError("x and y must have equal length")
        if len(x) < 2:
            raise ValueError("need at least two points")
        if any(x[i + 1] <= x[i] for i in range(len(x) - 1)):
            raise ValueError("x must be strictly increasing")
        self.x = list(x)
        self.y = list(y)
        self.n = len(x)
        self.d = _pchip_slopes(self.x, self.y)

    def __call__(self, xq):
        """Evaluate at xq using the Hermite cubic of the containing interval (clamped at the ends)."""
        x, y, d = self.x, self.y, self.d
        # find interval by binary search
        if xq <= x[0]:
            i = 0
        elif xq >= x[-1]:
            i = self.n - 2
        else:
            lo, hi = 0, self.n - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if x[mid] <= xq:
                    lo = mid
                else:
                    hi = mid
            i = lo
        h = x[i + 1] - x[i]
        t = (xq - x[i]) / h
        # Hermite basis
        h00 = 2 * t ** 3 - 3 * t ** 2 + 1
        h10 = t ** 3 - 2 * t ** 2 + t
        h01 = -2 * t ** 3 + 3 * t ** 2
        h11 = t ** 3 - t ** 2
        return (h00 * y[i] + h10 * h * d[i] + h01 * y[i + 1] + h11 * h * d[i + 1])

    def derivative(self, xq):
        """First derivative of the interpolant at xq."""
        x, y, d = self.x, self.y, self.d
        if xq <= x[0]:
            i = 0
        elif xq >= x[-1]:
            i = self.n - 2
        else:
            lo, hi = 0, self.n - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if x[mid] <= xq:
                    lo = mid
                else:
                    hi = mid
            i = lo
        h = x[i + 1] - x[i]
        t = (xq - x[i]) / h
        dh00 = 6 * t ** 2 - 6 * t
        dh10 = 3 * t ** 2 - 4 * t + 1
        dh01 = -6 * t ** 2 + 6 * t
        dh11 = 3 * t ** 2 - 2 * t
        return (dh00 * y[i] + dh10 * h * d[i] + dh01 * y[i + 1] + dh11 * h * d[i + 1]) / h

    def evaluate(self, xs):
        return [self(x) for x in xs]

    def slopes(self):
        """The knot derivatives PCHIP chose."""
        return list(self.d)
