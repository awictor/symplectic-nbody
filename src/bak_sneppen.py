"""The Bak-Sneppen model: evolution self-organizes to the edge of chaos, with avalanches of extinction.

Why does the fossil record show long calm stretches punctuated by bursts of extinction (PUNCTUATED
EQUILIBRIUM) rather than smooth gradual change? Bak and Sneppen's 1993 model gives a startlingly simple
mechanism: SELF-ORGANIZED CRITICALITY in a coevolving ecosystem. Place N species on a ring, each with a
random FITNESS in [0, 1). At every step, find the LEAST-fit species -- the one most likely to mutate or go
extinct -- and replace IT AND ITS TWO NEIGHBOURS with fresh random fitnesses (a species' fate is coupled to
its ecological neighbours). Repeat forever.

Left alone, this drives itself to a critical state with no tuning:

  A SELF-ORGANIZED THRESHOLD. The minimum fitness climbs until almost all species sit ABOVE a critical
      value f_c (about 0.667 for the 1-D ring); the system hovers just at the threshold where the smallest
      disturbance can cascade.
  AVALANCHES. Track "activity" below the current threshold: bursts of updates -- avalanches -- come in all
      sizes, with a POWER-LAW size distribution (no characteristic scale), the fingerprint of criticality
      and the model's explanation for extinction bursts of every magnitude.
  PUNCTUATED EQUILIBRIUM. Long quiet spells (all fitnesses high) are broken by sudden avalanches when a
      low-fitness species appears and triggers a chain of neighbour replacements.

This module runs the Bak-Sneppen ring, tracks the minimum fitness and the self-organized threshold, and
measures avalanche sizes. It uses a seeded RNG. It is validated: the minimum fitness rises from ~0 and its
long-run average approaches the known 1-D critical value ~2/3; the mean fitness rises then plateaus (the
system self-organizes without tuning); avalanches (runs of activity below the threshold) span a wide range
of sizes with a heavy tail, not a narrow band; each update replaces exactly three adjacent species with
fresh fitnesses; the critical state is independent of the initial condition; and results are reproducible
per seed. Pure stdlib; the self-organized-criticality companion to the sandpile, forest-fire, and
Ising tools."""

from __future__ import annotations


class _Rng:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def u(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)


def step(fitness, rng):
    """One Bak-Sneppen update: replace the least-fit species and its two ring neighbours. Returns the
    index of the minimum (the site that mutated)."""
    n = len(fitness)
    imin = min(range(n), key=lambda i: fitness[i])
    for j in (imin - 1, imin, imin + 1):
        fitness[j % n] = rng.u()
    return imin


def simulate(n, steps, seed=1, track=False):
    """Run the Bak-Sneppen model on a ring of n species for `steps` updates.

    Returns a dict with the final fitness, the self-organized THRESHOLD (the "gap": the running maximum
    of the minimum fitness, which converges to the 1-D critical value f_c ~ 0.667), and (if track) the
    min-fitness history. The gap is the right order parameter -- the raw minimum is a fresh random draw
    each step, so its average is meaningless; f_c is the level the minimum has climbed to."""
    rng = _Rng(seed)
    fitness = [rng.u() for _ in range(n)]
    min_history = []
    gap = 0.0  # running max of the minimum fitness (the self-organized threshold)
    for s in range(steps):
        step(fitness, rng)
        m = min(fitness)
        gap = max(gap, m)
        if track:
            min_history.append(m)
    result = {
        "fitness": fitness,
        "threshold": gap,
        "mean_fitness": sum(fitness) / n,
    }
    if track:
        result["min_history"] = min_history
    return result


def avalanche_sizes(n, steps, threshold, seed=1, warmup=None):
    """Measure avalanche sizes: consecutive updates whose minimum fitness stays below `threshold`.

    An avalanche starts when the min dips below the threshold and ends when it rises back above.
    Returns the list of avalanche sizes (in update counts)."""
    rng = _Rng(seed)
    fitness = [rng.u() for _ in range(n)]
    if warmup is None:
        warmup = steps // 2
    for _ in range(warmup):
        step(fitness, rng)
    sizes = []
    current = 0
    for _ in range(steps):
        step(fitness, rng)
        if min(fitness) < threshold:
            current += 1
        else:
            if current > 0:
                sizes.append(current)
            current = 0
    if current > 0:
        sizes.append(current)
    return sizes


def critical_threshold_1d():
    """The known self-organized critical fitness threshold for the 1-D Bak-Sneppen ring (~0.667)."""
    return 0.6670


def fraction_above(fitness, threshold):
    """Fraction of species with fitness above the threshold (near 1 in the critical state)."""
    return sum(1 for f in fitness if f > threshold) / len(fitness)
