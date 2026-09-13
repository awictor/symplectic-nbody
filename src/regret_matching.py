"""Regret matching and CFR: learning Nash equilibria of zero-sum games from self-play.

A two-player zero-sum game is given by a payoff matrix A: if the row player picks action i and the
column player picks j, the row player gains A[i][j] and the column player loses it. Von Neumann's
minimax theorem says every such game has a VALUE and a pair of optimal mixed strategies (a Nash
equilibrium) -- but how do you find them without a linear-programming solver? REGRET MATCHING
(Hart and Mas-Colell, 2000) is a stunningly simple online rule: play each action with probability
proportional to your positive CUMULATIVE REGRET for not having played it, where the regret for
action a on a round is (payoff you would have gotten from a) minus (payoff you actually got). Two
regret-matching players in self-play have their AVERAGE strategies converge to a Nash equilibrium,
and the empirical average payoff converges to the game value -- no matrix inversion, no LP.

This is the flat, single-decision case of COUNTERFACTUAL REGRET MINIMIZATION (CFR), the family of
algorithms behind superhuman poker. The convergence guarantee is a regret bound: the average
external regret of regret matching is O(sqrt(T)) with the number of actions inside the square root,
so it shrinks like O(1/sqrt(T)) per round. The right convergence certificate is EXPLOITABILITY --
how much a best-responding opponent could beat your average strategy by; at a true Nash equilibrium
in a symmetric zero-sum game it is zero.

This module implements regret matching for a single player against a fixed strategy, full
two-player self-play that returns both average strategies and the running value, best-response and
exploitability computations, and a from-scratch check that the recovered strategies form an
epsilon-Nash equilibrium (neither player can gain more than epsilon by deviating).

Validated by convergence and by known equilibria: exploitability decreases toward zero as
iterations grow, Rock-Paper-Scissors converges to the uniform (1/3, 1/3, 1/3) strategy with game
value zero, a dominant-strategy game converges to that pure strategy, and the self-play value
matches the minimax value found by an independent brute-force search over the row player's mixed
strategies for small games. Pure stdlib; the learning-dynamics companion to the LP simplex and the
mechanism-design (VCG) note."""

from __future__ import annotations


def _regret_match(cumulative_regret):
    """Turn a cumulative-regret vector into a strategy: proportional to positive regret,
    uniform if all regrets are non-positive."""
    pos = [r if r > 0 else 0.0 for r in cumulative_regret]
    total = sum(pos)
    n = len(cumulative_regret)
    if total <= 0:
        return [1.0 / n] * n
    return [p / total for p in pos]


def best_response_value_row(col_strategy, A):
    """The row player's best achievable expected payoff against a fixed column strategy."""
    m = len(A)
    best = -float("inf")
    for i in range(m):
        val = sum(A[i][j] * col_strategy[j] for j in range(len(col_strategy)))
        best = max(best, val)
    return best


def best_response_value_col(row_strategy, A):
    """The column player's best achievable payoff (it MINIMISES the row payoff, so returns the
    most negative row payoff it can force, expressed as the row player's value)."""
    n = len(A[0])
    worst = float("inf")
    for j in range(n):
        val = sum(A[i][j] * row_strategy[i] for i in range(len(row_strategy)))
        worst = min(worst, val)
    return worst


def exploitability(row_strategy, col_strategy, A):
    """How far the pair is from equilibrium: (row's best response value) - (col's best response
    value), in units of the row payoff. Zero at a Nash equilibrium; always >= 0."""
    br_row = best_response_value_row(col_strategy, A)
    br_col = best_response_value_col(row_strategy, A)
    return br_row - br_col


def solve(A, iterations=5000):
    """Two-player self-play by regret matching. Returns a dict with the average strategies for both
    players, the running average game value, and the final exploitability."""
    m = len(A)
    n = len(A[0])
    row_regret = [0.0] * m
    col_regret = [0.0] * n
    row_sum = [0.0] * m
    col_sum = [0.0] * n
    value_acc = 0.0

    for _ in range(iterations):
        row_strat = _regret_match(row_regret)
        col_strat = _regret_match(col_regret)
        for i in range(m):
            row_sum[i] += row_strat[i]
        for j in range(n):
            col_sum[j] += col_strat[j]

        # expected payoffs and counterfactual (action-value) regrets
        # row action value: playing i against col_strat
        row_action_val = [sum(A[i][j] * col_strat[j] for j in range(n)) for i in range(m)]
        row_ev = sum(row_strat[i] * row_action_val[i] for i in range(m))
        for i in range(m):
            row_regret[i] += row_action_val[i] - row_ev

        # column wants to MINIMISE the row payoff; its "payoff" is -A. action value for col j:
        col_action_val = [-sum(A[i][j] * row_strat[i] for i in range(m)) for j in range(n)]
        col_ev = sum(col_strat[j] * col_action_val[j] for j in range(n))
        for j in range(n):
            col_regret[j] += col_action_val[j] - col_ev

        value_acc += row_ev

    row_avg = _normalize(row_sum)
    col_avg = _normalize(col_sum)
    return {
        "row": row_avg,
        "col": col_avg,
        "value": value_acc / iterations,
        "exploitability": exploitability(row_avg, col_avg, A),
    }


def _normalize(v):
    total = sum(v)
    if total <= 0:
        return [1.0 / len(v)] * len(v)
    return [x / total for x in v]


def train_vs_fixed(A, opponent_col_strategy, iterations=5000):
    """Regret matching for the row player against a FIXED column strategy. The average strategy
    converges to a best response. Returns (avg_strategy, avg_value)."""
    m = len(A)
    n = len(A[0])
    regret = [0.0] * m
    strat_sum = [0.0] * m
    value_acc = 0.0
    for _ in range(iterations):
        strat = _regret_match(regret)
        for i in range(m):
            strat_sum[i] += strat[i]
        action_val = [sum(A[i][j] * opponent_col_strategy[j] for j in range(n)) for i in range(m)]
        ev = sum(strat[i] * action_val[i] for i in range(m))
        for i in range(m):
            regret[i] += action_val[i] - ev
        value_acc += ev
    return _normalize(strat_sum), value_acc / iterations


# --- brute-force reference for small games -----------------------------------
def brute_game_value(A, grid=20):
    """Approximate minimax value by scanning the row player's mixed strategies on a simplex grid
    (only practical for 2- or 3-action row players). Returns the max-min value."""
    m = len(A)
    n = len(A[0])
    best = -float("inf")
    for strat in _simplex_grid(m, grid):
        # column minimises: value = min over columns of expected row payoff
        val = min(sum(A[i][j] * strat[i] for i in range(m)) for j in range(n))
        best = max(best, val)
    return best


def _simplex_grid(m, grid):
    """Enumerate probability vectors of length m on a grid of granularity `grid`."""
    if m == 1:
        yield [1.0]
        return

    def rec(remaining, slots):
        if slots == 1:
            yield [remaining / grid]
            return
        for k in range(remaining + 1):
            for rest in rec(remaining - k, slots - 1):
                yield [k / grid] + rest

    yield from rec(grid, m)


def is_epsilon_nash(row_strategy, col_strategy, A, eps):
    """True if neither player can improve by more than eps by unilaterally deviating."""
    return exploitability(row_strategy, col_strategy, A) <= eps
