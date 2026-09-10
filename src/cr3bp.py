"""Circular Restricted Three-Body Problem (CR3BP) and its Lagrange points.

A massless test particle moves in the gravity of two massive bodies (primaries)
that orbit their common barycentre on a circle. Working in the *rotating frame*
that co-rotates with the primaries, the primaries sit still and the dynamics gain
centrifugal and Coriolis terms. In this frame there are five equilibrium points --
the Lagrange points L1..L5 -- and a conserved quantity, the Jacobi constant.

Nondimensional units: total mass = 1, separation = 1, angular rate = 1. The only
parameter is the mass ratio mu = m2 / (m1 + m2). The primaries sit at
x = -mu (mass 1-mu) and x = 1-mu (mass mu).

This is the mathematics behind halo orbits, Lagrange-point observatories (JWST at
Sun-Earth L2), and low-energy interplanetary transfers. Pure stdlib.
"""

from __future__ import annotations

import math
from typing import List, Tuple

Vec2 = Tuple[float, float]


class CR3BP:
    def __init__(self, mu: float):
        assert 0.0 < mu < 0.5
        self.mu = mu

    # --- distances to the two primaries ---------------------------------
    def _r1(self, x: float, y: float) -> float:
        return math.sqrt((x + self.mu) ** 2 + y * y)

    def _r2(self, x: float, y: float) -> float:
        return math.sqrt((x - (1.0 - self.mu)) ** 2 + y * y)

    def effective_potential(self, x: float, y: float) -> float:
        """Omega(x,y) = 1/2 (x^2 + y^2) + (1-mu)/r1 + mu/r2.
        Equilibria (Lagrange points) are the critical points of Omega."""
        mu = self.mu
        r1, r2 = self._r1(x, y), self._r2(x, y)
        return 0.5 * (x * x + y * y) + (1.0 - mu) / r1 + mu / r2

    def grad_omega(self, x: float, y: float) -> Vec2:
        """Gradient of the effective potential (equals zero at Lagrange points)."""
        mu = self.mu
        r1, r2 = self._r1(x, y), self._r2(x, y)
        r1c, r2c = r1 ** 3, r2 ** 3
        ox = x - (1.0 - mu) * (x + mu) / r1c - mu * (x - (1.0 - mu)) / r2c
        oy = y - (1.0 - mu) * y / r1c - mu * y / r2c
        return ox, oy

    def accel(self, state: List[float]) -> List[float]:
        """Rotating-frame equations of motion for state [x, y, vx, vy]:
            x'' = 2 vy + dOmega/dx
            y'' = -2 vx + dOmega/dy
        Returns [vx, vy, ax, ay]."""
        x, y, vx, vy = state
        ox, oy = self.grad_omega(x, y)
        ax = 2.0 * vy + ox
        ay = -2.0 * vx + oy
        return [vx, vy, ax, ay]

    def jacobi_constant(self, x: float, y: float, vx: float, vy: float) -> float:
        """C = 2 Omega - v^2. Conserved along any trajectory in the rotating frame."""
        return 2.0 * self.effective_potential(x, y) - (vx * vx + vy * vy)

    # --- Lagrange points -------------------------------------------------
    def collinear_points(self) -> Tuple[Vec2, Vec2, Vec2]:
        """L1, L2, L3 on the x-axis (y=0). Found by 1-D root finding on
        dOmega/dx along y=0 (bracketed by the primaries / far field)."""
        def gx(x):
            return self.grad_omega(x, 0.0)[0]

        mu = self.mu
        # brackets: L1 between primaries, L2 beyond m2, L3 beyond m1
        L1 = self._bisect(gx, -mu + 1e-4, 1.0 - mu - 1e-4)
        L2 = self._bisect(gx, 1.0 - mu + 1e-4, 2.0)
        L3 = self._bisect(gx, -2.0, -mu - 1e-4)
        return (L1, 0.0), (L2, 0.0), (L3, 0.0)

    def triangular_points(self) -> Tuple[Vec2, Vec2]:
        """L4, L5: exact equilateral-triangle points with the two primaries."""
        x = 0.5 - self.mu
        y = math.sqrt(3.0) / 2.0
        return (x, y), (x, -y)

    def lagrange_points(self):
        L1, L2, L3 = self.collinear_points()
        L4, L5 = self.triangular_points()
        return {"L1": L1, "L2": L2, "L3": L3, "L4": L4, "L5": L5}

    @staticmethod
    def _bisect(f, lo, hi, tol=1e-13, max_iter=200):
        flo, fhi = f(lo), f(hi)
        if flo == 0.0:
            return lo
        if fhi == 0.0:
            return hi
        if flo * fhi > 0.0:
            raise ValueError(f"root not bracketed on [{lo}, {hi}]: f={flo},{fhi}")
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            fm = f(mid)
            if abs(fm) < tol or (hi - lo) < tol:
                return mid
            if flo * fm < 0.0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm
        return 0.5 * (lo + hi)
