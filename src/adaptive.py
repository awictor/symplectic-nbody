"""Adaptive-step integration: Dormand-Prince RK45 with PI step-size control.

Fixed-step methods face a dilemma on eccentric orbits: the step that resolves a
fast pericenter passage is wastefully small out at apocenter. An *embedded*
Runge-Kutta pair produces two solutions of different order from the same stages;
their difference estimates the local error, which we use to grow or shrink the
step so the error stays near a user tolerance.

Dormand-Prince 5(4) is the pair behind MATLAB's ode45 and SciPy's RK45. It is
NOT symplectic -- energy will drift on very long runs -- but for getting a
high-accuracy trajectory over a bounded time with the fewest force evaluations,
it is the right tool, and it makes the fixed-vs-adaptive tradeoff concrete.
"""

from __future__ import annotations

import math
from typing import Callable, List, Tuple

Vec = List[float]

# Dormand-Prince 5(4) Butcher tableau.
_C = [0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0]
_A = [
    [],
    [1/5],
    [3/40, 9/40],
    [44/45, -56/15, 32/9],
    [19372/6561, -25360/2187, 64448/6561, -212/729],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
    [35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84],
]
# 5th-order solution weights (b) and 4th-order embedded weights (b*).
_B5 = [35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0]
_B4 = [5179/57600, 0.0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]


def _flatten(pos: List[Vec], vel: List[Vec]) -> List[float]:
    y = []
    for v in pos:
        y.extend(v)
    for v in vel:
        y.extend(v)
    return y


def _unflatten(y: List[float], n: int) -> Tuple[List[Vec], List[Vec]]:
    pos = [[y[3 * i], y[3 * i + 1], y[3 * i + 2]] for i in range(n)]
    off = 3 * n
    vel = [[y[off + 3 * i], y[off + 3 * i + 1], y[off + 3 * i + 2]] for i in range(n)]
    return pos, vel


class DormandPrince:
    """Adaptive RK45 driver over the first-order system y' = f(y) where
    y = (positions, velocities) and f = (velocities, accelerations)."""

    def __init__(self, accel: Callable[[List[Vec]], List[Vec]], n: int,
                 rtol: float = 1e-9, atol: float = 1e-12,
                 h_init: float = 1e-3, safety: float = 0.9,
                 min_scale: float = 0.2, max_scale: float = 5.0):
        self.accel = accel
        self.n = n
        self.rtol = rtol
        self.atol = atol
        self.h = h_init
        self.safety = safety
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.n_accepted = 0
        self.n_rejected = 0
        self.n_feval = 0

    def _deriv(self, y: List[float]) -> List[float]:
        pos, vel = _unflatten(y, self.n)
        acc = self.accel(pos)
        self.n_feval += 1
        dy = []
        for v in vel:
            dy.extend(v)
        for a in acc:
            dy.extend(a)
        return dy

    def _error_norm(self, y: List[float], y5: List[float], y4: List[float]) -> float:
        # RMS of (err / (atol + rtol*|y|)); target is <= 1.
        total = 0.0
        for i in range(len(y)):
            sc = self.atol + self.rtol * max(abs(y[i]), abs(y5[i]))
            e = (y5[i] - y4[i]) / sc
            total += e * e
        return math.sqrt(total / len(y))

    def _step_try(self, y: List[float], h: float):
        k = [None] * 7
        k[0] = self._deriv(y)
        for s in range(1, 7):
            ys = list(y)
            for i in range(len(y)):
                acc = 0.0
                for j in range(s):
                    acc += _A[s][j] * k[j][i]
                ys[i] += h * acc
            k[s] = self._deriv(ys)
        y5 = list(y)
        y4 = list(y)
        for i in range(len(y)):
            s5 = s4 = 0.0
            for s in range(7):
                s5 += _B5[s] * k[s][i]
                s4 += _B4[s] * k[s][i]
            y5[i] += h * s5
            y4[i] += h * s4
        return y5, y4

    def integrate(self, pos: List[Vec], vel: List[Vec], t_end: float,
                  on_sample=None):
        """Integrate from t=0 to t_end with adaptive steps. Optionally call
        on_sample(t, pos, vel) after each accepted step. Returns final (pos,vel)."""
        y = _flatten(pos, vel)
        t = 0.0
        h = min(self.h, t_end)
        while t < t_end:
            if t + h > t_end:
                h = t_end - t
            y5, y4 = self._step_try(y, h)
            err = self._error_norm(y, y5, y4)
            if err <= 1.0:
                # accept
                t += h
                y = y5
                self.n_accepted += 1
                if on_sample is not None:
                    p, v = _unflatten(y, self.n)
                    on_sample(t, p, v)
                # grow step (5th-order error ~ h^5 -> exponent 1/5)
                scale = self.safety * (err ** -0.2 if err > 0 else self.max_scale)
                h *= min(self.max_scale, scale)
            else:
                # reject, shrink, retry
                self.n_rejected += 1
                scale = self.safety * err ** -0.2
                h *= max(self.min_scale, scale)
            self.h = h
        return _unflatten(y, self.n)
