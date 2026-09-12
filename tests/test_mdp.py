"""Tests for mdp: VI/PI agreement, Bellman optimality, gridworld policy, discount effect."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mdp import MDP, gridworld

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


# --- value iteration and policy iteration agree ----------------------------
g = gridworld(4, 3, goal=(3, 2), obstacles=[(1, 1)], slip=0.0, gamma=0.9)
V_vi, pol_vi, _ = g.value_iteration()
V_pi, pol_pi, _ = g.policy_iteration()
check("VI and PI reach the same policy", pol_vi == pol_pi)
check("VI and PI reach the same values",
      all(abs(V_vi[s] - V_pi[s]) < 1e-6 for s in g.states))

# --- Bellman optimality: the value function is a fixed point ---------------
check("value iteration result satisfies Bellman optimality (zero residual)",
      g.bellman_residual(V_vi) < 1e-8)
check("policy iteration result satisfies Bellman optimality",
      g.bellman_residual(V_pi) < 1e-8)

# --- values reflect proximity to the goal ----------------------------------
# the goal is absorbing (no actions), so V(goal)=0; the states that REACH the goal have the
# highest value, and the far corner (many discounted steps away) has a lower value.
check("a near-goal state has the highest value", V_vi[(2, 2)] >= max(V_vi[s] for s in g.states) - 1e-9)
check("value decreases with distance (far corner < near-goal state)", V_vi[(0, 0)] < V_vi[(2, 2)])

# --- a deterministic gridworld: policy follows a shortest path -------------
g2 = gridworld(5, 1, goal=(4, 0), slip=0.0, gamma=0.95)   # a corridor
V2, pol2, _ = g2.value_iteration()
check("corridor policy always points East toward the goal",
      all(pol2[(x, 0)] == "E" for x in range(4)))

# --- higher discount values distant rewards more ---------------------------
def corner_value(gamma):
    gg = gridworld(6, 1, goal=(5, 0), slip=0.0, gamma=gamma, step_reward=-0.01, goal_reward=1.0)
    V, _, _ = gg.value_iteration()
    return V[(0, 0)]

check("higher discount gives the far corner a higher value",
      corner_value(0.99) > corner_value(0.5))

# --- a tiny hand-built MDP -------------------------------------------------
# two states: A and B. In A, action 'go' -> B with reward 10; in B, 'stay' -> B with reward 0.
# gamma small so the one-time reward dominates.
states = ["A", "B"]
transitions = {
    "A": {"go": [(1.0, "B", 10.0)], "wait": [(1.0, "A", 0.0)]},
    "B": {"stay": [(1.0, "B", 0.0)]},
}
mdp = MDP(states, ["go", "wait", "stay"], transitions, gamma=0.5)
V, pol, _ = mdp.value_iteration()
check("hand MDP: optimal action in A is 'go'", pol["A"] == "go")
check("hand MDP: V(A) = 10 (reward) + gamma*V(B) = 10", abs(V["A"] - 10.0) < 1e-6)
check("hand MDP: V(B) = 0", abs(V["B"]) < 1e-6)

# --- stochastic dynamics (slip) still solvable -----------------------------
gslip = gridworld(4, 4, goal=(3, 3), obstacles=[(1, 1), (2, 2)], slip=0.2, gamma=0.9)
Vs, pols, _ = gslip.value_iteration()
Vsp, polsp, _ = gslip.policy_iteration()
check("stochastic gridworld: VI and PI values agree",
      all(abs(Vs[s] - Vsp[s]) < 1e-5 for s in gslip.states))
check("stochastic gridworld satisfies Bellman optimality", gslip.bellman_residual(Vs) < 1e-7)

# --- value iteration error contracts by gamma each sweep -------------------
# track the per-sweep delta and confirm it shrinks roughly geometrically
class TrackMDP(MDP):
    def value_iteration_deltas(self, sweeps=30):
        V = {s: 0.0 for s in self.states}
        deltas = []
        for _ in range(sweeps):
            delta = 0.0
            newV = {}
            for s in self.states:
                acts = list(self.P[s].keys())
                if not acts:
                    newV[s] = 0.0
                    continue
                best = max(self._q_value(s, a, V) for a in acts)
                newV[s] = best
                delta = max(delta, abs(best - V[s]))
            V = newV
            deltas.append(delta)
        return deltas


# a self-looping MDP gives a smooth geometric contraction (not finite-step convergence):
# one state with a self-loop of reward 1; V converges to 1/(1-gamma), delta shrinks by exactly gamma.
loop = TrackMDP(["s"], ["go"], {"s": {"go": [(1.0, "s", 1.0)]}}, gamma=0.9)
loop_deltas = loop.value_iteration_deltas(sweeps=30)
ratios = [loop_deltas[i + 1] / loop_deltas[i] for i in range(3, 20) if loop_deltas[i] > 1e-12]
check("value-iteration error contracts by exactly the discount factor",
      ratios and all(abs(r - 0.9) < 1e-6 for r in ratios))

# --- convergence is fast for a reasonable discount -------------------------
_, _, vi_iters = g.value_iteration()
_, _, pi_iters = g.policy_iteration()
check("policy iteration converges in few rounds", pi_iters <= 10)
check("value iteration converges", vi_iters < 500)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all mdp tests passed")
