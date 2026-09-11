"""Parrondo's paradox: two losing games that together win.

Take two gambling games, each a losing proposition on its own -- play either forever and your
capital drifts to zero. Parrondo's paradox (Juan Parrondo, 1996) is that alternating between
them, or even choosing which to play at random each round, can make your capital drift UP. Order
from chaos: two ratchets that each slip backward, combined, climb.

The classic construction:

  * Game A is a single biased coin: win $1 with probability p = 1/2 - eps, lose $1 otherwise.
    A tiny eps > 0 makes it slightly losing.

  * Game B is capital-dependent: if your current capital is a multiple of 3 you flip a very bad
    coin (win prob p1 = 1/10 - eps); otherwise a good one (win prob p2 = 3/4 - eps). Tuned so
    that B, left alone, spends enough time on the bad coin to lose overall.

Each game is a random walk whose long-run drift is the stationary average of (2 w_s - 1) over
the capital-mod-3 states s, weighted by how often the walk sits in each state. Game B's losing
comes from the bad state being visited disproportionately; mixing in game A reshuffles that
occupancy so the good coin is flipped more often, and the combined drift turns positive. The
same flashing-ratchet mechanism drives molecular motors, models directed transport from noise,
and appears in evolutionary and financial models.

This module computes each game's per-round win probabilities, the exact stationary distribution
of the capital-mod-3 Markov chain, the resulting long-run drift (gain per round), and a seeded
Monte-Carlo capital trajectory, confirming that A and B lose alone while the mixture wins. Pure
stdlib; the counterintuitive-probability companion to the gambler's-ruin and Polya-walk notes.
"""

from __future__ import annotations

MOD = 3  # game B switches coin on capital divisible by this


def game_a_winprobs(eps: float = 0.005):
    """Win probabilities of game A by capital-mod-3 state: a flat coin 1/2 - eps everywhere."""
    p = 0.5 - eps
    return [p, p, p]


def game_b_winprobs(eps: float = 0.005):
    """Win probabilities of game B by capital-mod-3 state: bad coin (1/10 - eps) when capital is
    a multiple of 3, good coin (3/4 - eps) otherwise."""
    bad = 0.1 - eps
    good = 0.75 - eps
    return [bad, good, good]


def mixed_winprobs(eps: float = 0.005, gamma: float = 0.5):
    """Win probabilities when each round is game A with probability gamma, else game B.
    The per-state win probability is the gamma-blend of the two games' coins."""
    a = game_a_winprobs(eps)
    b = game_b_winprobs(eps)
    return [gamma * a[s] + (1.0 - gamma) * b[s] for s in range(MOD)]


def stationary_distribution(winprobs, iters: int = 20000, tol: float = 1e-14):
    """Stationary distribution of the capital-mod-3 Markov chain for the given per-state win
    probabilities. From state s the walk moves to (s+1) mod 3 on a win (prob w_s) and to
    (s-1) mod 3 on a loss. Solved by power iteration on the transition matrix."""
    m = MOD
    # transition matrix P[s][s']
    P = [[0.0] * m for _ in range(m)]
    for s in range(m):
        w = winprobs[s]
        P[s][(s + 1) % m] += w
        P[s][(s - 1) % m] += 1.0 - w
    pi = [1.0 / m] * m
    for _ in range(iters):
        new = [0.0] * m
        for s in range(m):
            for t in range(m):
                new[t] += pi[s] * P[s][t]
        diff = sum(abs(new[i] - pi[i]) for i in range(m))
        pi = new
        if diff < tol:
            break
    total = sum(pi)
    return [x / total for x in pi]


def drift(winprobs) -> float:
    """Long-run expected capital change per round: sum_s pi_s (2 w_s - 1), where pi is the
    stationary distribution. Negative = losing, positive = winning."""
    pi = stationary_distribution(winprobs)
    return sum(pi[s] * (2.0 * winprobs[s] - 1.0) for s in range(MOD))


def game_a_drift(eps: float = 0.005) -> float:
    """Long-run drift of game A alone (should be slightly negative for eps > 0)."""
    return drift(game_a_winprobs(eps))


def game_b_drift(eps: float = 0.005) -> float:
    """Long-run drift of game B alone (tuned to be slightly negative)."""
    return drift(game_b_winprobs(eps))


def mixed_drift(eps: float = 0.005, gamma: float = 0.5) -> float:
    """Long-run drift of the randomized A/B mixture (the paradox: positive though A, B < 0)."""
    return drift(mixed_winprobs(eps, gamma))


class _Rng:
    """Seeded LCG; high bits (an LCG's low bits are not random)."""

    def __init__(self, seed: int = 1):
        self.state = seed & 0xFFFFFFFF

    def random(self) -> float:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)


def _play_round(rng, capital, game, eps, gamma):
    """One round; `game` is 'A', 'B', 'mix', or 'AABB' (a deterministic period-4 pattern is
    handled by the caller). Returns the capital change +1 or -1."""
    if game == "A":
        w = 0.5 - eps
    elif game == "B":
        w = (0.1 - eps) if capital % MOD == 0 else (0.75 - eps)
    else:  # random mixture
        if rng.random() < gamma:
            w = 0.5 - eps
        else:
            w = (0.1 - eps) if capital % MOD == 0 else (0.75 - eps)
    return 1 if rng.random() < w else -1


def simulate(game: str, rounds: int = 100000, eps: float = 0.005, gamma: float = 0.5,
             seed: int = 1) -> float:
    """Monte-Carlo mean capital change per round playing `game` ('A', 'B', or 'mix') for
    `rounds` rounds from capital 0. Returns the realized drift (final capital / rounds)."""
    rng = _Rng(seed)
    capital = 0
    for _ in range(rounds):
        capital += _play_round(rng, capital, game, eps, gamma)
    return capital / rounds


def simulate_trajectory(game: str, rounds: int, eps: float = 0.005, gamma: float = 0.5,
                        seed: int = 1, every: int = 1):
    """Capital trajectory: list of capital values sampled every `every` rounds (including the
    start), for plotting A vs B vs the mixture."""
    rng = _Rng(seed)
    capital = 0
    traj = [0]
    for t in range(1, rounds + 1):
        capital += _play_round(rng, capital, game, eps, gamma)
        if t % every == 0:
            traj.append(capital)
    return traj
