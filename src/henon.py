"""The Henon map: a strange attractor you can hold in two lines.

Michel Henon built this map in 1976 as a stripped-down model of the stretching-and-folding
that makes weather chaotic, simple enough to iterate by hand yet showing a genuine strange
attractor. It takes a point in the plane to a new one by

    x_{n+1} = 1 - a x_n^2 + y_n
    y_{n+1} = b x_n,

and at the classic parameters a = 1.4, b = 0.3 the iterates never settle and never repeat:
they trace out the Henon attractor, a fractal curve that looks like a set of nested arcs and,
zoomed in, reveals ever-finer parallel strands -- a Cantor-set cross-section. It is the
canonical low-dimensional strange attractor.

The map is dissipative: it contracts areas by the constant Jacobian determinant

    |J| = |b|,

so a blob of initial points shrinks to zero area (onto the attractor) while being stretched
and folded, giving a fractal of dimension ~1.26 (between a line and a plane). Along the way
the largest Lyapunov exponent is positive (~0.42 nat/iteration for the classic parameters):
nearby points separate exponentially, the signature of chaos. Change the parameters and the
same map period-doubles into and out of chaos just like the logistic map, its 1D cousin.

This module gives one map step, the trajectory and the attractor point cloud, the area-
contraction factor, the two fixed points and their existence condition, and an estimate of
the largest Lyapunov exponent, and reproduces the |b| area contraction and the positive
Lyapunov exponent of the classic attractor. Pure stdlib; the strange-attractor companion to
the logistic-map and N-body Lyapunov notes.
"""

from __future__ import annotations

import math

A_CLASSIC = 1.4
B_CLASSIC = 0.3


def step(x: float, y: float, a: float = A_CLASSIC, b: float = B_CLASSIC):
    """One Henon iteration: (x, y) -> (1 - a x^2 + y, b x)."""
    return 1.0 - a * x * x + y, b * x


def trajectory(x0: float, y0: float, n: int, a: float = A_CLASSIC, b: float = B_CLASSIC):
    """The first n iterates as a list of (x, y) points, starting from (x0, y0)."""
    pts = [(x0, y0)]
    x, y = x0, y0
    for _ in range(n):
        x, y = step(x, y, a, b)
        pts.append((x, y))
    return pts


def attractor_points(n: int, a: float = A_CLASSIC, b: float = B_CLASSIC,
                     transient: int = 200):
    """n points on the attractor, after discarding a transient so the orbit has settled onto
    it. Returns a list of (x, y)."""
    x, y = 0.0, 0.0
    for _ in range(transient):
        x, y = step(x, y, a, b)
    pts = []
    for _ in range(n):
        x, y = step(x, y, a, b)
        pts.append((x, y))
    return pts


def area_contraction(b: float = B_CLASSIC) -> float:
    """Area-contraction factor per iteration, |det J| = |b|. <1 means the map is dissipative,
    shrinking any blob of initial conditions onto the (lower-dimensional) attractor."""
    return abs(b)


def fixed_points(a: float = A_CLASSIC, b: float = B_CLASSIC):
    """The map's fixed points (x*, y* = b x*), solving a x^2 + (1-b) x - 1 = 0. Returns a list
    of (x, y) (two points when the discriminant is positive, else empty)."""
    disc = (1.0 - b) ** 2 + 4.0 * a
    if disc < 0.0 or a == 0.0:
        return []
    root = math.sqrt(disc)
    xs = [(-(1.0 - b) + root) / (2.0 * a), (-(1.0 - b) - root) / (2.0 * a)]
    return [(x, b * x) for x in xs]


def largest_lyapunov(a: float = A_CLASSIC, b: float = B_CLASSIC,
                     n: int = 20000, transient: int = 500) -> float:
    """Estimate of the largest Lyapunov exponent (nat/iteration) by evolving a tangent vector
    under the Jacobian [[-2 a x, 1], [b, 0]] and averaging the log stretch. Positive => chaos;
    ~0.42 for the classic attractor."""
    x, y = 0.0, 0.0
    for _ in range(transient):
        x, y = step(x, y, a, b)
    vx, vy = 1.0, 0.0
    total = 0.0
    for _ in range(n):
        # tangent map
        nvx = -2.0 * a * x * vx + vy
        nvy = b * vx
        norm = math.sqrt(nvx * nvx + nvy * nvy)
        if norm > 0.0:
            total += math.log(norm)
            vx, vy = nvx / norm, nvy / norm
        x, y = step(x, y, a, b)
    return total / n


def is_chaotic(a: float = A_CLASSIC, b: float = B_CLASSIC, **kwargs) -> bool:
    """True if the largest Lyapunov exponent is positive -- the orbit is chaotic (a strange
    attractor rather than a periodic cycle)."""
    return largest_lyapunov(a, b, **kwargs) > 0.0
