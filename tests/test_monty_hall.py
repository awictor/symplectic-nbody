"""Tests for monty_hall.py -- the Monty Hall problem.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Exact probabilities are checked
against known values and a seeded Monte-Carlo play.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import monty_hall as m  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


# --- classic 3-door game ----------------------------------------------------
check("staying wins 1/3", approx(m.stay_probability(3), 1 / 3, 1e-12))
check("switching wins 2/3", approx(m.switch_probability(3, 1), 2 / 3, 1e-12))
check("switching is exactly twice as good", approx(m.switch_advantage(3, 1), 2.0, 1e-12))
check("stay + switch = 1 in the classic game",
      approx(m.stay_probability(3) + m.switch_probability(3, 1), 1.0, 1e-12))
check("switching always beats staying", m.switch_probability(3, 1) > m.stay_probability(3))

# --- informed vs random host -----------------------------------------------
check("random (knowledge-free) host makes switching only 1/2",
      approx(m.random_host_switch_probability(3), 0.5, 1e-12))
check("the knowing host gives a strictly better switch than the random host",
      m.switch_probability(3, 1) > m.random_host_switch_probability(3))

# --- generalized N doors, k reveals ----------------------------------------
check("N=10 stay is 1/10", approx(m.stay_probability(10), 0.1, 1e-12))
check("N=10, 1 reveal switch is (1/8)(9/10)", approx(m.switch_probability(10, 1), (1 / 8) * 9 / 10, 1e-12))
# opening all but one other door (k = N-2) makes switching nearly certain
check("N=10, 8 reveals switch is 9/10", approx(m.switch_probability(10, 8), 0.9, 1e-12))
check("N=100, 98 reveals switch is 99/100", approx(m.switch_probability(100, 98), 0.99, 1e-12))
check("switching beats staying for every N", all(
    m.switch_probability(N, 1) > m.stay_probability(N) for N in (3, 5, 10, 50)))
check("the switch advantage grows with more reveals",
      m.switch_probability(10, 8) > m.switch_probability(10, 1))

# --- input validation -------------------------------------------------------
try:
    m.stay_probability(2)
    check("rejects fewer than 3 doors", False)
except ValueError:
    check("rejects fewer than 3 doors", True)
try:
    m.switch_probability(5, 4)  # reveals must be <= N-2 = 3
    check("rejects too many reveals", False)
except ValueError:
    check("rejects too many reveals", True)
try:
    m.switch_probability(5, 0)
    check("rejects zero reveals", False)
except ValueError:
    check("rejects zero reveals", True)

# --- Monte-Carlo agreement --------------------------------------------------
check("simulated stay matches 1/3",
      approx(m.simulate("stay", trials=40000, seed=5), 1 / 3, 0.02))
check("simulated switch matches 2/3",
      approx(m.simulate("switch", trials=40000, seed=5), 2 / 3, 0.02))
check("simulated random-host switch matches 1/2",
      approx(m.simulate("switch", trials=60000, seed=5, informed_host=False), 0.5, 0.02))
for N, k in ((10, 1), (10, 8), (100, 98)):
    check(f"simulated switch matches theory (N={N}, k={k})",
          approx(m.simulate("switch", N, k, trials=40000, seed=3), m.switch_probability(N, k), 0.02))
# staying is unaffected by the number of doors revealed
check("simulated stay in a big game is still 1/N",
      approx(m.simulate("stay", 10, 8, trials=40000, seed=3), 0.1, 0.02))


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall monty_hall tests passed")
