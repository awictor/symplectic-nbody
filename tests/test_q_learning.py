"""Tests for q_learning: converges to the MDP optimum, learns faster than random, SARSA works."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from q_learning import GridEnv, q_learning, sarsa, greedy_rollout, _RNG
from mdp import gridworld

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


GOAL = (3, 2)
OBST = [(1, 1)]


def make_env(seed=1):
    return GridEnv(4, 3, goal=GOAL, obstacles=OBST, slip=0.0, seed=seed)


# ground truth from value iteration
g = gridworld(4, 3, goal=GOAL, obstacles=OBST, slip=0.0, gamma=0.9)
V_opt, pol_opt, _ = g.value_iteration()

# --- Q-learning converges to the optimal policy ----------------------------
env = make_env()
Q, pol, returns = q_learning(env, episodes=1500, alpha=0.5, gamma=0.9, epsilon=0.3, seed=2)
non_goal = [s for s in env.states if s != GOAL]
matches = sum(1 for s in non_goal if pol[s] == pol_opt[s])
check(f"Q-learning greedy policy matches value iteration ({matches}/{len(non_goal)})",
      matches == len(non_goal))

# --- the greedy rollout reaches the goal on the optimal-length path --------
opt_env = make_env()
opt_steps = greedy_rollout(opt_env, pol_opt)
ql_env = make_env()
ql_steps = greedy_rollout(ql_env, pol)
check(f"Q-learning reaches the goal in the optimal number of steps ({ql_steps} == {opt_steps})",
      ql_steps == opt_steps)

# --- learned Q-values approach the MDP optimal values ----------------------
# V(s) = max_a Q(s,a); compare to VI values on the optimal path
def learned_value(Q, s):
    return max(Q[s].values())


# check a few states on the way to the goal
close_states = [(2, 2), (3, 1), (2, 1)]
val_ok = all(abs(learned_value(Q, s) - V_opt[s]) < 0.15 for s in close_states if s in Q)
check("learned Q-values approximate the MDP optimal values near the goal", val_ok)

# --- a learned agent beats a random walker ---------------------------------
def random_walk_steps(env, seed, max_steps=200):
    rng = _RNG(seed)
    s = env.reset()
    for t in range(1, max_steps + 1):
        a = env.actions[rng.randint(len(env.actions))]
        s, r, done = env.step(a)
        if done:
            return t
    return max_steps


rand_total = sum(random_walk_steps(make_env(seed=s), seed=s + 100) for s in range(10)) / 10
learned_total = greedy_rollout(make_env(), pol)
check(f"learned policy reaches the goal far faster than random ({learned_total} vs {rand_total:.0f})",
      learned_total < rand_total)

# --- returns improve over training -----------------------------------------
early = sum(returns[:100]) / 100
late = sum(returns[-100:]) / 100
check(f"episode return improves with training ({early:.2f} -> {late:.2f})", late > early)

# --- SARSA also solves the task --------------------------------------------
senv = make_env()
Qs, pol_s, ret_s = sarsa(senv, episodes=1500, alpha=0.5, gamma=0.9, epsilon=0.2, seed=3)
sarsa_steps = greedy_rollout(make_env(), pol_s)
check(f"SARSA reaches the goal efficiently ({sarsa_steps} steps)", sarsa_steps <= opt_steps + 2)
sarsa_early = sum(ret_s[:100]) / 100
sarsa_late = sum(ret_s[-100:]) / 100
check("SARSA return improves with training", sarsa_late > sarsa_early)

# --- a larger open gridworld ------------------------------------------------
big = GridEnv(6, 6, goal=(5, 5), obstacles=[(2, 2), (3, 3), (1, 4)], slip=0.0, seed=1)
gbig = gridworld(6, 6, goal=(5, 5), obstacles=[(2, 2), (3, 3), (1, 4)], slip=0.0, gamma=0.9)
_, pol_big_opt, _ = gbig.value_iteration()
Qb, pol_big, _ = q_learning(big, episodes=3000, alpha=0.5, gamma=0.9, epsilon=0.3, seed=5)
# the rollout should reach the goal in the optimal number of steps
opt_big = greedy_rollout(GridEnv(6, 6, goal=(5, 5), obstacles=[(2, 2), (3, 3), (1, 4)], seed=1), pol_big_opt)
ql_big = greedy_rollout(GridEnv(6, 6, goal=(5, 5), obstacles=[(2, 2), (3, 3), (1, 4)], seed=1), pol_big)
check(f"Q-learning solves a 6x6 gridworld optimally ({ql_big} == {opt_big})", ql_big == opt_big)

# --- higher learning rate speeds early learning ----------------------------
_, _, ret_slow = q_learning(make_env(), episodes=200, alpha=0.05, epsilon=0.2, seed=7)
_, _, ret_fast = q_learning(make_env(), episodes=200, alpha=0.7, epsilon=0.2, seed=7)
check("higher learning rate learns faster early on",
      sum(ret_fast[:80]) > sum(ret_slow[:80]))

# --- reproducibility -------------------------------------------------------
_, p1, r1 = q_learning(make_env(), episodes=100, seed=9)
_, p2, r2 = q_learning(make_env(), episodes=100, seed=9)
check("same seed gives identical returns", r1 == r2)

# --- Q-values are non-trivial (learning happened) --------------------------
check("some Q-values are positive after training", any(max(Q[s].values()) > 0 for s in env.states))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all q_learning tests passed")
