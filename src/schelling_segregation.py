"""Schelling's segregation model: how mild individual preferences produce total collective segregation.

Thomas Schelling's 1971 model is the founding example of EMERGENCE in social science, and it delivers a
genuinely counterintuitive result: a population where every individual is perfectly happy to live in a
mixed neighbourhood -- demanding only that, say, a THIRD of their neighbours share their type -- will
nonetheless self-organize into sharply SEGREGATED blocks. No one wants segregation; it emerges anyway from
the tiny local preference.

The setup: a grid holds agents of two types plus empty cells. An agent is UNHAPPY if the fraction of its
occupied neighbours that share its type falls below a tolerance THRESHOLD tau. Each round, unhappy agents
move to a random empty cell. Iterating until everyone is happy (or no improving move exists), the grid's
average SIMILARITY -- the fraction of neighbours sharing one's type -- climbs far above tau: with tau = 1/3
the equilibrium similarity is typically ~0.7-0.8, deep segregation from a mild wish. The model is the
canonical demonstration that macro-patterns need not mirror micro-motives.

This module builds a random mixed grid, computes each agent's local similarity and happiness, moves unhappy
agents to empty cells until equilibrium, and reports the global segregation (mean similarity) and the
happy fraction. It uses a seeded RNG. It is validated: from a random start the mean similarity RISES far
above the tolerance threshold (segregation emerges from mild preference); the happy fraction increases
toward 1 as the dynamics run; a higher tolerance threshold produces MORE segregation; the number of empty
cells and the count of each type are conserved by moves; tau = 0 leaves the grid essentially unsegregated
(everyone always happy); and results are reproducible per seed. Pure stdlib; the agent-based-emergence
companion to the voter-model, Ising, and cellular-automaton tools."""

from __future__ import annotations

EMPTY = 0


class _Rng:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def u(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)

    def randint(self, lo, hi):
        return lo + int(self.u() * (hi - lo))

    def shuffle(self, seq):
        for i in range(len(seq) - 1, 0, -1):
            j = self.randint(0, i + 1)
            seq[i], seq[j] = seq[j], seq[i]


def make_grid(rows, cols, empty_frac, seed=1):
    """Random grid: each cell empty (0) with prob empty_frac, else type 1 or 2 with equal odds."""
    rng = _Rng(seed)
    cells = []
    for _ in range(rows * cols):
        if rng.u() < empty_frac:
            cells.append(EMPTY)
        else:
            cells.append(1 if rng.u() < 0.5 else 2)
    return [cells[r * cols:(r + 1) * cols] for r in range(rows)], rng


def _neighbors(grid, r, c):
    rows, cols = len(grid), len(grid[0])
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                out.append(grid[nr][nc])
    return out


def similarity(grid, r, c):
    """Fraction of an agent's OCCUPIED neighbours that share its type (1.0 if no occupied neighbours)."""
    me = grid[r][c]
    if me == EMPTY:
        return 0.0
    neigh = [v for v in _neighbors(grid, r, c) if v != EMPTY]
    if not neigh:
        return 1.0
    same = sum(1 for v in neigh if v == me)
    return same / len(neigh)


def is_happy(grid, r, c, tau):
    """An occupied agent is happy if its similarity >= tolerance tau."""
    return similarity(grid, r, c) >= tau


def mean_similarity(grid):
    """Average similarity over all occupied cells -- the global segregation measure."""
    rows, cols = len(grid), len(grid[0])
    total = 0.0
    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != EMPTY:
                total += similarity(grid, r, c)
                count += 1
    return total / count if count else 0.0


def happy_fraction(grid, tau):
    """Fraction of occupied agents that are happy."""
    rows, cols = len(grid), len(grid[0])
    happy = 0
    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != EMPTY:
                count += 1
                if is_happy(grid, r, c, tau):
                    happy += 1
    return happy / count if count else 1.0


def _empty_cells(grid):
    rows, cols = len(grid), len(grid[0])
    return [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == EMPTY]


def step(grid, tau, rng):
    """One round: every unhappy agent (in random order) moves to a random empty cell. Returns #moves."""
    rows, cols = len(grid), len(grid[0])
    unhappy = [(r, c) for r in range(rows) for c in range(cols)
               if grid[r][c] != EMPTY and not is_happy(grid, r, c, tau)]
    rng.shuffle(unhappy)
    moves = 0
    for r, c in unhappy:
        if grid[r][c] == EMPTY:
            continue  # already vacated this round
        empties = _empty_cells(grid)
        if not empties:
            break
        er, ec = empties[rng.randint(0, len(empties))]
        grid[er][ec] = grid[r][c]
        grid[r][c] = EMPTY
        moves += 1
    return moves


def simulate(rows, cols, empty_frac, tau, max_rounds=100, seed=1, track=False):
    """Run Schelling to equilibrium. Returns a dict with the final grid, rounds, final mean_similarity,
    happy_fraction, and (if track) the per-round similarity/happiness history."""
    grid, rng = make_grid(rows, cols, empty_frac, seed=seed)
    history = [(mean_similarity(grid), happy_fraction(grid, tau))]
    rounds = 0
    for _ in range(max_rounds):
        moves = step(grid, tau, rng)
        rounds += 1
        if track:
            history.append((mean_similarity(grid), happy_fraction(grid, tau)))
        if moves == 0:
            break
    result = {"grid": grid, "rounds": rounds,
              "mean_similarity": mean_similarity(grid),
              "happy_fraction": happy_fraction(grid, tau),
              "initial_similarity": history[0][0]}
    if track:
        result["history"] = history
    return result


def type_counts(grid):
    """Count of empty, type-1, and type-2 cells."""
    flat = [v for row in grid for v in row]
    return flat.count(EMPTY), flat.count(1), flat.count(2)
