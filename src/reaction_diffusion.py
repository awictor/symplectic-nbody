"""Reaction-diffusion: how a chemical soup grows spots and stripes.

Alan Turing's 1952 insight was that patterns -- a leopard's spots, a zebra's stripes, ridges on
a seashell -- can arise spontaneously from two chemicals that react and diffuse, with no
blueprint. If an "activator" that promotes itself diffuses slowly while an "inhibitor" that
suppresses it diffuses fast, a uniform mixture becomes unstable and settles into a standing
pattern. The Gray-Scott model is the classic two-species version:

    du/dt = Du * lap(u) - u v^2 + F (1 - u)
    dv/dt = Dv * lap(v) + u v^2 - (F + k) v,

where u is the substrate, v the autocatalyst, F the feed rate, k the kill rate, Du > Dv the
diffusion coefficients, and lap the Laplacian (spatial diffusion). The reaction u v^2 is
autocatalytic: v makes more v by consuming u. Depending on (F, k) the same equations produce
spots, stripes, mazes, self-replicating blobs, or travelling waves -- one of the richest pattern
zoos in all of applied math, and a working model of biological morphogenesis.

This module runs Gray-Scott on a grid: the 5-point Laplacian, one Euler time step, evolution
from a seeded perturbation, and the pattern's total-v measure, and reproduces the growth of
structure from a small seed and the stability of a uniform steady state. Pure stdlib (seeded
LCG for the initial noise); the pattern-formation companion to the cellular-automaton and
diffusion notes.
"""

from __future__ import annotations


class _Rng:
    """Seeded LCG; high bits."""

    def __init__(self, seed: int = 1):
        self.state = seed & 0xFFFFFFFF

    def random(self) -> float:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)


def make_fields(n: int):
    """Initial u (=1) and v (=0) fields on an n x n grid: the substrate-full, autocatalyst-
    empty steady state before any seed."""
    u = [[1.0 for _ in range(n)] for _ in range(n)]
    v = [[0.0 for _ in range(n)] for _ in range(n)]
    return u, v


def seed_center(u, v, size: int = 6):
    """Seed a small central square with autocatalyst (v=1, u=0.5) to break the symmetry and
    let a pattern grow."""
    n = len(u)
    c = n // 2
    for r in range(c - size // 2, c + size // 2):
        for cc in range(c - size // 2, c + size // 2):
            if 0 <= r < n and 0 <= cc < n:
                u[r][cc] = 0.5
                v[r][cc] = 0.25
    return u, v


def laplacian(field, r, c):
    """5-point Laplacian at (r,c) with periodic (toroidal) boundaries:
    sum of 4 neighbours minus 4 times the centre."""
    n = len(field)
    return (field[(r - 1) % n][c] + field[(r + 1) % n][c]
            + field[r][(c - 1) % n] + field[r][(c + 1) % n]
            - 4.0 * field[r][c])


def step(u, v, du=0.16, dv=0.08, f=0.035, k=0.065, dt=1.0):
    """One explicit Euler Gray-Scott step, returning new (u, v). Du>Dv (activator-inhibitor);
    the u v^2 reaction is autocatalytic in v."""
    n = len(u)
    nu = [[0.0] * n for _ in range(n)]
    nv = [[0.0] * n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            uvv = u[r][c] * v[r][c] * v[r][c]
            nu[r][c] = u[r][c] + dt * (du * laplacian(u, r, c) - uvv + f * (1.0 - u[r][c]))
            nv[r][c] = v[r][c] + dt * (dv * laplacian(v, r, c) + uvv - (f + k) * v[r][c])
    return nu, nv


def evolve(u, v, steps, **kw):
    """Advance the fields `steps` Gray-Scott steps; return the final (u, v)."""
    for _ in range(steps):
        u, v = step(u, v, **kw)
    return u, v


def total_v(v) -> float:
    """Total amount of autocatalyst on the grid: 0 for a bare substrate, growing as a pattern
    forms."""
    return sum(sum(row) for row in v)


def pattern_contrast(field) -> float:
    """Max minus min of a field: ~0 for a uniform state, large once a pattern has formed
    (spots/stripes span a range of concentration)."""
    flat = [x for row in field for x in row]
    return max(flat) - min(flat)
