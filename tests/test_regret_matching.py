"""Tests for regret matching / CFR: exploitability -> 0, known equilibria, minimax value match."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from regret_matching import (  # noqa: E402
    solve,
    train_vs_fixed,
    exploitability,
    best_response_value_row,
    brute_game_value,
    is_epsilon_nash,
)


PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def main():
    # ---- 1. Rock-Paper-Scissors converges to uniform, value 0 ---------------------------
    # A[i][j]: row payoff. R>S? standard RPS: win +1, lose -1, tie 0.
    #        R    P    S
    #  R [   0,  -1,  +1 ]
    #  P [  +1,   0,  -1 ]
    #  S [  -1,  +1,   0 ]
    rps = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
    res = solve(rps, iterations=20000)
    row = res["row"]
    check("RPS row strategy ~ uniform",
          all(abs(p - 1 / 3) < 0.03 for p in row), f"{[round(p,3) for p in row]}")
    check("RPS game value ~ 0", abs(res["value"]) < 0.03, f"{res['value']:.4f}")
    check("RPS exploitability small", res["exploitability"] < 0.03, f"{res['exploitability']:.4f}")

    # ---- 2. exploitability decreases with iterations ------------------------------------
    # a larger asymmetric game converges more slowly, so the trend is visible
    big = [[2, -1, 0, 3], [-1, 3, 1, -2], [0, 1, -2, 2], [3, -2, 2, -1]]
    lo = solve(big, iterations=20)["exploitability"]
    hi = solve(big, iterations=20000)["exploitability"]
    check("exploitability shrinks with more iterations", hi < lo, f"{lo:.4f} -> {hi:.4f}")

    # ---- 3. dominant strategy game: row always prefers action 0 -------------------------
    # row action 0 dominates action 1 (better payoff in every column)
    dom = [[3, 2], [1, 0]]
    res = solve(dom, iterations=5000)
    check("dominant strategy: row picks action 0", res["row"][0] > 0.97, f"{res['row']}")
    # column: action 1 gives row {2,0}, action 0 gives row {3,1}; column minimises -> picks col 1
    check("dominant strategy: col picks action 1", res["col"][1] > 0.97, f"{res['col']}")
    check("dominant game value ~ 2", abs(res["value"] - 2) < 0.05, f"{res['value']:.4f}")

    # ---- 4. matching pennies converges, value 0 ----------------------------------------
    pennies = [[1, -1], [-1, 1]]
    res = solve(pennies, iterations=20000)
    check("matching pennies row ~ (0.5,0.5)",
          abs(res["row"][0] - 0.5) < 0.03, f"{res['row']}")
    check("matching pennies value ~ 0", abs(res["value"]) < 0.03, f"{res['value']:.4f}")

    # ---- 5. self-play value matches brute-force minimax for small games -----------------
    games = [
        rps,
        pennies,
        [[3, 2], [1, 0]],
        [[2, -1, 0], [-1, 3, 1], [0, 1, -2]],
        [[4, 0, 2], [1, 3, 2]],  # 2x3
    ]
    max_err = 0.0
    for A in games:
        cfr_val = solve(A, iterations=20000)["value"]
        brute = brute_game_value(A, grid=40)
        max_err = max(max_err, abs(cfr_val - brute))
    check("self-play value ~ brute minimax (5 games)", max_err < 0.06, f"max err {max_err:.4f}")

    # ---- 6. epsilon-Nash certificate ----------------------------------------------------
    res = solve(rps, iterations=20000)
    check("recovered RPS pair is epsilon-Nash (eps=0.05)",
          is_epsilon_nash(res["row"], res["col"], rps, 0.05))
    check("exploitability always non-negative",
          exploitability(res["row"], res["col"], rps) >= -1e-9)

    # ---- 7. train_vs_fixed converges to a best response ---------------------------------
    # against a fixed column playing pure "Paper" (col index 1), row should learn "Scissors" (i=2)
    fixed_col = [0.0, 1.0, 0.0]
    strat, val = train_vs_fixed(rps, fixed_col, iterations=5000)
    check("best response to pure Paper is Scissors", strat[2] > 0.97, f"{[round(p,3) for p in strat]}")
    check("best response value ~ 1 (Scissors beats Paper)", abs(val - 1) < 0.05, f"{val:.4f}")
    # and it should equal the analytic best-response value
    check("train_vs_fixed value ~ best_response_value",
          abs(val - best_response_value_row(fixed_col, rps)) < 0.05)

    # ---- 8. degenerate / edge cases -----------------------------------------------------
    # a 1x1 game: only one action each, value is the single entry
    res = solve([[5]], iterations=100)
    check("1x1 game value = entry", abs(res["value"] - 5) < 1e-9)
    check("1x1 strategies are singletons", res["row"] == [1.0] and res["col"] == [1.0])

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
