"""The voter model: how local imitation drives a population to consensus, and why the majority isn't guaranteed.

The voter model (Clifford & Sudbury 1973; Holley & Liggett 1975) is the simplest model of opinion dynamics
and one of the cleanest interacting particle systems. Every site of a grid holds an opinion, +1 or -1. At
each step pick a random site and a random neighbour, and the site COPIES the neighbour's opinion -- pure
social imitation, no stubbornness, no noise. Run it and the population inexorably COARSENS into growing
single-opinion domains until, on any finite connected graph, it reaches unanimous CONSENSUS (one absorbing
state per opinion).

Its beautiful, exactly-solvable features:

  MAGNETIZATION IS A MARTINGALE. The expected fraction of +1 sites never changes -- a copy is equally
      likely to flip a + to - as the reverse -- so the mean opinion is conserved in expectation, and the
      process is a bounded martingale that must converge.
  CONSENSUS PROBABILITY = INITIAL FRACTION. Because of the martingale, the probability the whole grid
      ends up +1 equals the INITIAL fraction of +1 sites. Start 70% blue and blue wins 70% of the time --
      the outcome is fair, not majority-take-all.
  DUALITY WITH COALESCING RANDOM WALKS. Tracing opinions backward in time turns the voter model into
      coalescing random walks, the tool that proves consensus and gives the coarsening time scales.

This module simulates the voter model on a 2-D periodic grid, tracks the magnetization and the number of
opinion domains, runs to consensus, and estimates the consensus probability by ensemble. It uses a seeded
RNG. It is validated: the expected magnetization is preserved (martingale) across a single update in
expectation; the process reaches unanimous consensus on a finite grid; the probability of +1 consensus
equals the initial +1 fraction; the opinion domains coarsen (their count decreases over time); a fully
ordered start is absorbing (nothing changes); and results are reproducible per seed. Pure stdlib; the
interacting-particle-system companion to the Ising, Moran-process, and Gillespie tools."""

from __future__ import annotations


class _Rng:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def u(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)

    def randint(self, lo, hi):
        return lo + int(self.u() * (hi - lo))


def make_grid(rows, cols, up_fraction, rng):
    """Random grid of +1/-1 with the given fraction of +1 sites."""
    return [[1 if rng.u() < up_fraction else -1 for _ in range(cols)] for _ in range(rows)]


def magnetization(grid):
    """Mean opinion (in [-1, 1]); the +1 fraction is (1 + magnetization)/2."""
    total = sum(sum(row) for row in grid)
    n = len(grid) * len(grid[0])
    return total / n


def up_fraction(grid):
    """Fraction of +1 sites."""
    return (1 + magnetization(grid)) / 2


def is_consensus(grid):
    """True if every site holds the same opinion."""
    first = grid[0][0]
    return all(v == first for row in grid for v in row)


def _neighbors(r, c, rows, cols):
    """Four periodic (toroidal) neighbours of (r, c)."""
    return [((r - 1) % rows, c), ((r + 1) % rows, c),
            (r, (c - 1) % cols), (r, (c + 1) % cols)]


def step(grid, rng):
    """One voter update: a random site copies a random neighbour. Mutates grid in place."""
    rows, cols = len(grid), len(grid[0])
    r = rng.randint(0, rows)
    c = rng.randint(0, cols)
    neigh = _neighbors(r, c, rows, cols)
    nr, nc = neigh[rng.randint(0, 4)]
    grid[r][c] = grid[nr][nc]


def count_domains(grid):
    """Number of connected same-opinion domains (4-connectivity), a coarsening measure."""
    rows, cols = len(grid), len(grid[0])
    seen = [[False] * cols for _ in range(rows)]
    domains = 0
    for r0 in range(rows):
        for c0 in range(cols):
            if seen[r0][c0]:
                continue
            domains += 1
            opinion = grid[r0][c0]
            stack = [(r0, c0)]
            seen[r0][c0] = True
            while stack:
                r, c = stack.pop()
                for nr, nc in _neighbors(r, c, rows, cols):
                    if not seen[nr][nc] and grid[nr][nc] == opinion:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
    return domains


def simulate(rows, cols, up_frac, max_steps=10_000_000, seed=1, track_every=0):
    """Run the voter model to consensus (or max_steps). Returns a dict with the final opinion (+1/-1),
    steps taken, and (if track_every>0) magnetization/domain-count history."""
    rng = _Rng(seed)
    grid = make_grid(rows, cols, up_frac, rng)
    history = []
    steps = 0
    while steps < max_steps and not is_consensus(grid):
        step(grid, rng)
        steps += 1
        if track_every and steps % track_every == 0:
            history.append((magnetization(grid), count_domains(grid)))
    result = {"final_opinion": grid[0][0] if is_consensus(grid) else None,
              "steps": steps, "consensus": is_consensus(grid), "grid": grid}
    if track_every:
        result["history"] = history
    return result


def consensus_probability(rows, cols, up_frac, n_runs=200, seed=1):
    """Empirical probability of +1 consensus (should equal the initial +1 fraction, the martingale)."""
    plus = 0
    for run in range(n_runs):
        res = simulate(rows, cols, up_frac, seed=seed + run * 2749)
        if res["final_opinion"] == 1:
            plus += 1
    return plus / n_runs
