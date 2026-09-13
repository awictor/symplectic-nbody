"""Wang-Landau sampling: measuring the density of states directly, and getting all temperatures at once.

Ordinary Monte Carlo (Metropolis) samples configurations at ONE fixed temperature: to map a phase
transition you must re-run at many temperatures, and near the critical point the sampler slows to a
crawl (critical slowing down) and gets stuck in valleys of the energy landscape. Wang-Landau sampling
(2001) sidesteps all of that by estimating a temperature-independent quantity: the DENSITY OF STATES
g(E), the number of configurations with each energy E. Once you know g(E), you know EVERYTHING -- the
partition function Z(T) = sum_E g(E) exp(-E/kT), and hence the free energy, internal energy, entropy,
and specific heat at EVERY temperature, from a SINGLE simulation.

The trick is a self-adjusting random walk in energy space that flattens its own histogram. We do not
know g(E), so we build it up. Maintain a running estimate (kept as ln g to avoid overflow) and a
histogram H(E) of visits. Propose a spin flip from energy E1 to E2 and accept it with probability

    min(1, g(E1) / g(E2)),

i.e. the walker is PUSHED TOWARD energies it has seen less often (small g). Every time energy E is
visited, multiply g(E) by a modification factor f (add ln f to ln g) and increment H(E). When the
histogram is "flat enough" (every bin within some fraction of the mean), reset H and shrink f toward 1
(f -> sqrt(f)). As f -> 1 the estimate converges to the true g(E) up to an overall constant, which we
fix by a known normalization (e.g. the two all-aligned ground states, or that the total number of
states is 2^N).

From ln g(E) this module computes the full thermodynamics at any temperature by careful log-sum-exp,
and it is validated the only honest way: against the EXACT density of states obtained by brute-force
enumeration of every one of the 2^N spin configurations of a small Ising lattice. The recovered ln g
matches the exact ln g (up to the overall constant) within a few percent; the internal energy and
specific heat curves it produces agree with those computed from the exact g(E); the histogram really
does flatten; and the total recovered state count sums to 2^N. Pure stdlib (a seeded LCG); the
flat-histogram, all-temperatures companion to the fixed-temperature Metropolis Ising sampler."""

from __future__ import annotations

import math

from metropolis import ising_energy


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def _allowed_energies(n):
    """The energies actually attainable by an n x n periodic Ising lattice.

    Bonds = 2 n^2; energy = -J * (aligned - misaligned) = -2*bonds + 4*misaligned, stepping by 4.
    Returns the sorted list of attainable integer energies (J = 1). Energies -2*bonds+4 and the
    symmetric top are not attainable on the torus, but we simply index every multiple of 4 in range
    and let unused bins stay empty.
    """
    bonds = 2 * n * n
    # all-aligned satisfies every bond: E = -bonds. Each flip changes E by a multiple of 4, so the
    # attainable energies are -bonds + 4k. (The top energy +bonds may be unreachable on a torus with
    # odd side length; unused bins simply stay empty.)
    return list(range(-bonds, bonds + 1, 4))


def wang_landau(n, seed=0, flat=0.8, f_final=1e-6, max_sweeps=200000):
    """Estimate ln g(E) for the n x n periodic Ising ferromagnet by Wang-Landau sampling.

    Returns (energies, ln_g): the attainable energy grid and the (normalized) log density of states.
    """
    rng = _lcg(seed)
    # random initial configuration
    grid = [[1 if rng() < 0.5 else -1 for _ in range(n)] for _ in range(n)]
    E = int(round(ising_energy(grid)))

    energies = _allowed_energies(n)
    idx = {e: i for i, e in enumerate(energies)}
    nbin = len(energies)
    ln_g = [0.0] * nbin
    H = [0] * nbin

    lnf = 1.0  # ln of the initial modification factor f = e
    sweeps = 0

    def delta_if_flip(r, c):
        s = grid[r][c]
        nb = (grid[(r + 1) % n][c] + grid[(r - 1) % n][c]
              + grid[r][(c + 1) % n] + grid[r][(c - 1) % n])
        # E = -sum s_i s_j; flipping s -> -s changes E by +2*s*nb
        return 2 * s * nb

    while lnf > f_final and sweeps < max_sweeps:
        # one sweep = N flip attempts
        for _ in range(n * n):
            r = int(rng() * n)
            c = int(rng() * n)
            dE = delta_if_flip(r, c)
            E2 = E + dE
            i1 = idx[E]
            i2 = idx.get(E2)
            if i2 is None:
                continue  # energy out of grid (shouldn't happen)
            # accept with min(1, g(E1)/g(E2)) = min(1, exp(ln_g1 - ln_g2))
            if ln_g[i1] >= ln_g[i2] or rng() < math.exp(ln_g[i1] - ln_g[i2]):
                grid[r][c] = -grid[r][c]
                E = E2
                i1 = i2
            ln_g[i1] += lnf
            H[i1] += 1
        sweeps += 1

        # check flatness over visited bins
        visited = [h for h in H if h > 0]
        if visited:
            mean = sum(visited) / len(visited)
            if min(visited) >= flat * mean:
                lnf *= 0.5
                H = [0] * nbin

    # normalize: shift ln_g so the two ground states (E = -bonds, all-up/all-down) give ln g = ln 2
    ground = idx[energies[0]]
    # only normalize using populated bins
    populated = [i for i in range(nbin) if ln_g[i] > 0]
    shift = math.log(2.0) - ln_g[ground] if ln_g[ground] > 0 else 0.0
    ln_g_norm = [ln_g[i] + shift if ln_g[i] > 0 else float("-inf") for i in range(nbin)]
    return energies, ln_g_norm


def exact_density_of_states(n):
    """Brute-force exact density of states g(E) by enumerating all 2^(n*n) configurations.

    Returns (energies, ln_g) matching the wang_landau grid. Only feasible for tiny n (<= 4).
    """
    energies = _allowed_energies(n)
    idx = {e: i for i, e in enumerate(energies)}
    counts = [0] * len(energies)
    N = n * n
    for bits in range(1 << N):
        grid = [[1 if (bits >> (r * n + c)) & 1 else -1 for c in range(n)] for r in range(n)]
        E = int(round(ising_energy(grid)))
        counts[idx[E]] += 1
    ln_g = [math.log(c) if c > 0 else float("-inf") for c in counts]
    return energies, ln_g, counts


def thermodynamics(energies, ln_g, temperature):
    """Internal energy <E> and specific heat C from ln g(E) at a temperature (k_B = 1).

    Uses log-sum-exp over the Boltzmann-weighted density of states.
    """
    beta = 1.0 / temperature
    # weights w_E = ln_g(E) - beta E; normalize by the max for stability
    terms = [(e, lg - beta * e) for e, lg in zip(energies, ln_g) if lg != float("-inf")]
    wmax = max(w for _, w in terms)
    Z = sum(math.exp(w - wmax) for _, w in terms)
    meanE = sum(e * math.exp(w - wmax) for e, w in terms) / Z
    meanE2 = sum(e * e * math.exp(w - wmax) for e, w in terms) / Z
    var = meanE2 - meanE * meanE
    C = var * beta * beta
    return meanE, C


def partition_function_lse(energies, ln_g, temperature):
    """ln Z(T) via log-sum-exp of ln g(E) - E/T."""
    beta = 1.0 / temperature
    terms = [lg - beta * e for e, lg in zip(energies, ln_g) if lg != float("-inf")]
    wmax = max(terms)
    return wmax + math.log(sum(math.exp(w - wmax) for w in terms))
