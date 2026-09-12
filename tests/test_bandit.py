"""Tests for bandit: policies beat random, sublinear regret, best-arm ID, log growth."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bandit import (BernoulliBandit, epsilon_greedy, ucb1, thompson_sampling, random_policy)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


PROBS = [0.2, 0.5, 0.75, 0.4, 0.6]
BEST = 2


def run(fn, rounds=2000, seed=7, **kw):
    b = BernoulliBandit(PROBS, seed=seed)
    return b, fn(b, rounds, **kw)


# --- environment sanity ----------------------------------------------------
b = BernoulliBandit(PROBS, seed=1)
check("best arm identified by the environment", b.best == BEST)
pulls = [b.pull(2) for _ in range(5000)]
check("arm 2 pays out near its probability 0.75", abs(sum(pulls) / 5000 - 0.75) < 0.05)

# --- every learning policy beats random ------------------------------------
_, (r_rand, reg_rand, _, _) = run(random_policy)
for name, fn in [("epsilon_greedy", epsilon_greedy), ("ucb1", ucb1), ("thompson", thompson_sampling)]:
    _, (rew, reg, counts, means) = run(fn)
    check(f"{name} beats random on total reward", sum(rew) > sum(r_rand) * 1.2)
    check(f"{name} has lower regret than random", reg[-1] < reg_rand[-1])

# --- best-arm identification: the best arm is the most-pulled --------------
for name, fn in [("epsilon_greedy", epsilon_greedy), ("ucb1", ucb1), ("thompson", thompson_sampling)]:
    _, (rew, reg, counts, means) = run(fn)
    most_pulled = max(range(len(counts)), key=lambda i: counts[i])
    check(f"{name} pulls the best arm most often", most_pulled == BEST)

# --- UCB1 and Thompson achieve sublinear regret ----------------------------
# average regret per round should fall as rounds grow
for name, fn in [("ucb1", ucb1), ("thompson", thompson_sampling)]:
    b1, (_, reg_short, _, _) = run(fn, rounds=500)
    b2, (_, reg_long, _, _) = run(fn, rounds=5000)
    avg_short = reg_short[-1] / 500
    avg_long = reg_long[-1] / 5000
    check(f"{name} average regret decreases with more rounds ({avg_short:.3f} -> {avg_long:.3f})",
          avg_long < avg_short)

# --- UCB1 regret grows sub-linearly (logarithmically) ----------------------
# compare cumulative regret at t and 4t: linear would ~4x, log would be ~ +const
b, (_, reg, _, _) = run(ucb1, rounds=8000)
r2000 = reg[1999]
r8000 = reg[7999]
check(f"UCB1 regret grows far slower than linearly ({r2000:.1f} -> {r8000:.1f}, not 4x)",
      r8000 < r2000 * 2.5)

# --- Thompson typically has the lowest regret ------------------------------
regs = {}
for name, fn in [("epsilon_greedy", epsilon_greedy), ("ucb1", ucb1), ("thompson", thompson_sampling)]:
    _, (_, reg, _, _) = run(fn, rounds=3000)
    regs[name] = reg[-1]
check(f"Thompson achieves competitive regret ({regs})", regs["thompson"] <= min(regs.values()) * 1.5)

# --- a single dominant arm is found quickly --------------------------------
easy = BernoulliBandit([0.1, 0.9, 0.1], seed=5)
_, reg, counts, means = ucb1(easy, 1000)
check("dominant arm gets the vast majority of pulls", counts[1] > 850)

# --- decaying epsilon-greedy improves over fixed ---------------------------
b, (_, reg_fixed, _, _) = run(epsilon_greedy, rounds=3000, epsilon=0.1, decay=False)
b, (_, reg_decay, _, _) = run(epsilon_greedy, rounds=3000, epsilon=1.0, decay=True)
check("decaying epsilon-greedy has lower regret than fixed",
      reg_decay[-1] < reg_fixed[-1])

# --- means converge to the true probabilities for pulled arms --------------
_, (_, _, counts, means) = run(thompson_sampling, rounds=5000)
# the best arm is pulled a lot, so its mean should be close to 0.75
check("estimated mean of the best arm converges to its probability",
      abs(means[BEST] - PROBS[BEST]) < 0.05)

# --- regret is monotonically non-decreasing --------------------------------
_, (_, reg, _, _) = run(ucb1, rounds=1000)
check("cumulative regret is non-decreasing", all(reg[i + 1] >= reg[i] - 1e-9 for i in range(len(reg) - 1)))

# --- reproducibility -------------------------------------------------------
_, out1 = run(thompson_sampling, rounds=500, seed=9)
_, out2 = run(thompson_sampling, rounds=500, seed=9)
check("same seed gives identical regret trajectory", out1[1] == out2[1])

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all bandit tests passed")
