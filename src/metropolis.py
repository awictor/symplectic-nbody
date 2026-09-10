"""Metropolis Monte Carlo: sampling the Boltzmann distribution by biased coin flips.

Most many-body systems cannot be summed exactly -- the 2D Ising model has 2^(N) states -- so we
sample them. The Metropolis algorithm (1953) is the workhorse: propose a small change, and
accept it with probability

    P(accept) = min(1, exp(-dE / (k_B T))),

where dE is the energy change. Downhill moves are always taken; uphill moves are taken with a
Boltzmann-weighted chance. This simple rule satisfies detailed balance, so the chain of states
it visits is distributed exactly as exp(-E/kT) -- letting you measure thermal averages by just
averaging over the walk.

Run it on the 2D Ising ferromagnet (spins +/-1 on a grid, each favouring alignment with its
four neighbours) and it reproduces the real phase transition that mean-field theory only
approximates: below the Onsager critical temperature

    T_c = 2 J / (k_B ln(1 + sqrt(2)))  ~  2.269 J / k_B,

the spins order into a magnetized state; above it they are disordered. Unlike mean field, the
Metropolis simulation captures the true fluctuations and the exact 2D T_c.

This module gives the Metropolis acceptance rule, the Ising energy and magnetization of a
configuration, a sweep of single-spin updates, the exact 2D critical temperature, and a full
run returning the average magnetization at a temperature, and reproduces the accept-all-
downhill rule and the ordered/disordered transition across T_c. Pure stdlib (seeded LCG,
high bits; no random module); the sampling companion to the mean-field Ising and percolation
notes.
"""

from __future__ import annotations

import math

ONSAGER_TC = 2.0 / math.log(1.0 + math.sqrt(2.0))   # 2D Ising T_c in units of J/k_B ~ 2.269


class _Rng:
    """Seeded LCG; high bits (low bits of these constants are non-random)."""

    def __init__(self, seed: int = 1):
        self.state = seed & 0xFFFFFFFF

    def random(self) -> float:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)

    def randint(self, k: int) -> int:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 16) % k


def accept_probability(delta_e: float, temperature: float) -> float:
    """Metropolis acceptance probability min(1, exp(-dE/T)) (k_B = 1). 1 for downhill moves
    (dE <= 0); Boltzmann-suppressed for uphill ones, more so at low temperature."""
    if delta_e <= 0.0:
        return 1.0
    return math.exp(-delta_e / temperature)


def ising_energy(grid, coupling: float = 1.0) -> float:
    """Total Ising energy E = -J sum over neighbour pairs s_i s_j (periodic boundaries).
    Lower when neighbours align."""
    n = len(grid)
    e = 0.0
    for r in range(n):
        for c in range(n):
            s = grid[r][c]
            e -= coupling * s * grid[(r + 1) % n][c]
            e -= coupling * s * grid[r][(c + 1) % n]
    return e


def magnetization(grid) -> float:
    """Mean spin (net magnetization per site) of the configuration, in [-1, 1]."""
    n = len(grid)
    return sum(sum(row) for row in grid) / (n * n)


def _local_delta(grid, r, c, coupling):
    """Energy change from flipping spin (r,c): dE = 2 J s (sum of 4 neighbours)."""
    n = len(grid)
    s = grid[r][c]
    nb = (grid[(r + 1) % n][c] + grid[(r - 1) % n][c]
          + grid[r][(c + 1) % n] + grid[r][(c - 1) % n])
    return 2.0 * coupling * s * nb


def sweep(grid, temperature: float, rng, coupling: float = 1.0):
    """One Metropolis sweep: attempt N*N single-spin flips, accepting by the Metropolis rule.
    Mutates grid in place."""
    n = len(grid)
    for _ in range(n * n):
        r, c = rng.randint(n), rng.randint(n)
        de = _local_delta(grid, r, c, coupling)
        if de <= 0.0 or rng.random() < math.exp(-de / temperature):
            grid[r][c] = -grid[r][c]


def run(n: int, temperature: float, sweeps: int = 200, coupling: float = 1.0, seed: int = 1):
    """Run Metropolis on an n x n Ising grid at temperature T, starting fully aligned. Returns
    the average |magnetization| over the second half of the run (after equilibration)."""
    rng = _Rng(seed)
    grid = [[1 for _ in range(n)] for _ in range(n)]
    mags = []
    for s in range(sweeps):
        sweep(grid, temperature, rng, coupling)
        if s >= sweeps // 2:
            mags.append(abs(magnetization(grid)))
    return sum(mags) / len(mags)
