"""Tests for parrondo.py -- Parrondo's paradox.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. The Markov-chain drift is
checked against exact values and a seeded Monte-Carlo run.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import parrondo as p  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


# --- stationary distribution -----------------------------------------------
piB = p.stationary_distribution(p.game_b_winprobs())
check("stationary distribution sums to 1", approx(sum(piB), 1.0, 1e-9))
check("stationary distribution is nonnegative", all(x >= 0 for x in piB))
check("stationary distribution has 3 states", len(piB) == 3)
# it must be a fixed point of the chain: verify pi P = pi for game B
w = p.game_b_winprobs()
m = p.MOD
P = [[0.0] * m for _ in range(m)]
for s in range(m):
    P[s][(s + 1) % m] += w[s]
    P[s][(s - 1) % m] += 1.0 - w[s]
piP = [sum(piB[s] * P[s][t] for s in range(m)) for t in range(m)]
check("stationary distribution is a fixed point (pi P = pi)",
      all(approx(piP[t], piB[t], 1e-6) for t in range(m)))
# game B over-visits the bad state (s=0) relative to uniform 1/3 -- that's why it loses
check("game B over-visits the bad (multiple-of-3) state", piB[0] > 1.0 / 3.0)

# a flat game (game A) has the uniform stationary distribution
piA = p.stationary_distribution(p.game_a_winprobs())
check("flat game A has ~uniform occupancy", all(approx(x, 1.0 / 3.0, 1e-6) for x in piA))

# --- the paradox: A and B lose, the mix wins -------------------------------
dA = p.game_a_drift()
dB = p.game_b_drift()
dMix = p.mixed_drift()
check("game A loses on its own (drift < 0)", dA < 0)
check("game B loses on its own (drift < 0)", dB < 0)
check("the random mixture wins (drift > 0)", dMix > 0)
check("the mixture beats both individual games", dMix > dA and dMix > dB)
# game A's drift is exactly -2 eps (flat coin, no state dependence)
check("game A drift is exactly -2 eps", approx(dA, -2 * 0.005, 1e-9))

# a fair epsilon = 0 makes both games break even (drift ~ 0) individually...
check("eps=0 makes game A break even", approx(p.game_a_drift(0.0), 0.0, 1e-9))
check("eps=0 makes game B break even", approx(p.game_b_drift(0.0), 0.0, 1e-6))
# ...yet the mixture STILL wins -- the purest form of the paradox: two exactly fair games
# combine into a winning one, because mixing reshapes the state occupancy.
check("eps=0 mixture wins even though both games are fair", p.mixed_drift(0.0) > 0)

# --- dependence on the mixing fraction -------------------------------------
# all-A (gamma=1) reduces to game A; all-B (gamma=0) reduces to game B
check("gamma=1 mixture equals game A", approx(p.mixed_drift(gamma=1.0), dA, 1e-9))
check("gamma=0 mixture equals game B", approx(p.mixed_drift(gamma=0.0), dB, 1e-9))
# an intermediate blend is where the paradox lives
check("a 50/50 blend is winning", p.mixed_drift(gamma=0.5) > 0)

# --- Monte-Carlo agreement --------------------------------------------------
simA = p.simulate("A", rounds=200000, seed=7)
simB = p.simulate("B", rounds=200000, seed=7)
simMix = p.simulate("mix", rounds=200000, seed=7)
check("simulated game A loses", simA < 0)
check("simulated game B loses", simB < 0)
check("simulated mixture wins", simMix > 0)
check("simulated mixture drift near theory", approx(simMix, dMix, 0.01))
check("simulated A drift near theory", approx(simA, dA, 0.01))

# --- trajectory -------------------------------------------------------------
traj = p.simulate_trajectory("mix", rounds=5000, seed=2, every=100)
check("trajectory starts at 0", traj[0] == 0)
check("trajectory has the expected sample count", len(traj) == 5000 // 100 + 1)
check("winning mixture trends upward over a long run",
      p.simulate_trajectory("mix", rounds=20000, seed=2)[-1] > 0)


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall parrondo tests passed")
