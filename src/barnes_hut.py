"""Barnes-Hut octree force solver: O(N log N) gravity.

Direct summation (nbody.NBody.accel) is O(N^2) — fine for a handful of bodies,
hopeless for thousands. Barnes-Hut groups distant bodies into their centre of
mass and treats the group as a single particle when it is "far enough away",
measured by the opening angle theta = s / d (cell size / distance).

theta = 0  -> never approximate, exact but O(N^2)
theta ~ 0.5 -> the classic accuracy/speed sweet spot
larger theta -> faster, less accurate

This module mirrors nbody's acceleration interface so it drops straight into
the same symplectic integrators.
"""

from __future__ import annotations

import math
from typing import List, Optional

Vec = List[float]


class _Cell:
    """One node of the octree: a cube covering [center-half, center+half]^3."""
    __slots__ = ("cx", "cy", "cz", "half", "mass", "comx", "comy", "comz",
                 "children", "body", "is_leaf")

    def __init__(self, cx: float, cy: float, cz: float, half: float):
        self.cx, self.cy, self.cz = cx, cy, cz
        self.half = half
        self.mass = 0.0
        self.comx = self.comy = self.comz = 0.0  # accumulated mass*position
        self.children: Optional[List[Optional["_Cell"]]] = None
        self.body: Optional[int] = None  # index of the single body in a leaf
        self.is_leaf = True

    def _octant(self, x: float, y: float, z: float) -> int:
        return (1 if x >= self.cx else 0) | (2 if y >= self.cy else 0) | (4 if z >= self.cz else 0)

    def _child_cell(self, octant: int) -> "_Cell":
        h = self.half * 0.5
        ox = h if (octant & 1) else -h
        oy = h if (octant & 2) else -h
        oz = h if (octant & 4) else -h
        return _Cell(self.cx + ox, self.cy + oy, self.cz + oz, h)


class BarnesHut:
    def __init__(self, G: float = 1.0, theta: float = 0.5, softening: float = 0.0):
        self.G = G
        self.theta2 = theta * theta
        self.soft2 = softening * softening

    def _build(self, pos: List[Vec], masses: List[float]) -> _Cell:
        # Root cube must enclose every body; size it from the coordinate span.
        xs = [p[0] for p in pos]; ys = [p[1] for p in pos]; zs = [p[2] for p in pos]
        cx = 0.5 * (min(xs) + max(xs))
        cy = 0.5 * (min(ys) + max(ys))
        cz = 0.5 * (min(zs) + max(zs))
        span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
        half = max(span * 0.5, 1e-12) * 1.0000001  # pad so boundary bodies land inside
        root = _Cell(cx, cy, cz, half)
        for i in range(len(pos)):
            self._insert(root, i, pos, masses)
        return root

    def _insert(self, cell: _Cell, i: int, pos: List[Vec], masses: List[float]) -> None:
        # Accumulate mass and mass-weighted position (finalized to COM at read time).
        m = masses[i]
        cell.mass += m
        cell.comx += m * pos[i][0]
        cell.comy += m * pos[i][1]
        cell.comz += m * pos[i][2]

        if cell.is_leaf and cell.body is None:
            cell.body = i
            return

        if cell.is_leaf and cell.body is not None:
            # Split: push the resident body down, then continue with the new one.
            cell.children = [None] * 8
            cell.is_leaf = False
            resident = cell.body
            cell.body = None
            self._place(cell, resident, pos, masses)

        self._place(cell, i, pos, masses)

    def _place(self, cell: _Cell, i: int, pos: List[Vec], masses: List[float]) -> None:
        oct_ = cell._octant(pos[i][0], pos[i][1], pos[i][2])
        if cell.children[oct_] is None:
            cell.children[oct_] = cell._child_cell(oct_)
        self._insert(cell.children[oct_], i, pos, masses)

    def _accel_on(self, cell: _Cell, x: float, y: float, z: float, i: int,
                  pos: List[Vec]) -> Vec:
        ax = ay = az = 0.0
        # Iterative traversal (explicit stack) to avoid deep recursion for big N.
        stack = [cell]
        while stack:
            c = stack.pop()
            if c.mass == 0.0:
                continue
            comx = c.comx / c.mass
            comy = c.comy / c.mass
            comz = c.comz / c.mass
            dx = comx - x; dy = comy - y; dz = comz - z
            r2 = dx * dx + dy * dy + dz * dz
            if c.is_leaf:
                if c.body == i or r2 == 0.0:
                    continue
                r2 += self.soft2
                inv_r3 = 1.0 / (r2 * math.sqrt(r2))
                f = self.G * c.mass * inv_r3
                ax += f * dx; ay += f * dy; az += f * dz
            else:
                # Opening criterion: (cell width)^2 / distance^2 < theta^2 -> approximate.
                width = 2.0 * c.half
                if (width * width) < self.theta2 * r2:
                    r2s = r2 + self.soft2
                    inv_r3 = 1.0 / (r2s * math.sqrt(r2s))
                    f = self.G * c.mass * inv_r3
                    ax += f * dx; ay += f * dy; az += f * dz
                else:
                    for ch in c.children:
                        if ch is not None:
                            stack.append(ch)
        return [ax, ay, az]

    def accel(self, pos: List[Vec], masses: List[float]) -> List[Vec]:
        root = self._build(pos, masses)
        return [self._accel_on(root, pos[i][0], pos[i][1], pos[i][2], i, pos)
                for i in range(len(pos))]
