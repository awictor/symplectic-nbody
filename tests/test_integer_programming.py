"""Tests for ILP branch-and-bound: matches brute force, integer+feasible, LP bound dominates, knapsack."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from integer_programming import (  # noqa: E402
    solve_ilp,
    lp_relaxation_bound,
    solve_knapsack_ilp,
    brute_ilp,
    _is_integer,
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


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def main():
    # ---- 1. classic textbook ILP -------------------------------------------------------
    # maximize 5x + 4y s.t. 6x+4y<=24, x+2y<=6, x,y>=0 integer.
    # LP optimum is (3, 1.5) = 21; the integer optimum is (4, 0) = 20 (6*4=24 binds, 4<=6).
    c = [5, 4]
    cons = [([6, 4], "<=", 24), ([1, 2], "<=", 6)]
    res = solve_ilp(c, cons, maximize=True)
    check("textbook ILP value = 20", res["status"] == "optimal" and abs(res["value"] - 20) < 1e-6,
          f"{res}")
    check("textbook ILP solution integer", all(_is_integer(v) for v in res["x"]))

    # ---- 2. LP relaxation bound >= integer optimum (maximize) ---------------------------
    lp = lp_relaxation_bound(c, cons, maximize=True)
    check("LP bound dominates ILP (maximize)", lp >= res["value"] - 1e-6, f"LP {lp} vs ILP {res['value']}")

    # ---- 3. matches brute force on random small ILPs ------------------------------------
    rng = _lcg(1)
    ok = True
    for trial in range(15):
        n = 2 + int(rng() * 2)  # 2-3 vars
        c = [1 + int(rng() * 9) for _ in range(n)]
        # a couple of <= constraints with positive coeffs
        cons = []
        for _ in range(2):
            a = [1 + int(rng() * 4) for _ in range(n)]
            b = 5 + int(rng() * 15)
            cons.append((a, "<=", b))
        res = solve_ilp(c, cons, maximize=True, var_bounds={j: (0, 10) for j in range(n)})
        bval, bx = brute_ilp(c, cons, True, [(0, 10)] * n)
        if res["value"] is None or abs(res["value"] - bval) > 1e-6:
            ok = False
            check("ILP == brute force", False, f"trial {trial}: {res['value']} vs {bval}")
            break
    if ok:
        check("ILP == brute force (15 random maximize ILPs)", True)

    # ---- 4. returned solution is feasible -----------------------------------------------
    c = [3, 2, 4]
    cons = [([1, 1, 1], "<=", 5), ([2, 1, 3], "<=", 8)]
    res = solve_ilp(c, cons, maximize=True, var_bounds={j: (0, 5) for j in range(3)})
    x = res["x"]
    check("solution satisfies constraints",
          sum(x) <= 5 + 1e-6 and 2 * x[0] + x[1] + 3 * x[2] <= 8 + 1e-6)
    check("solution is integer", all(_is_integer(v) for v in x))

    # ---- 5. 0/1 knapsack matches brute force --------------------------------------------
    values = [60, 100, 120]
    weights = [10, 20, 30]
    capacity = 50
    res = solve_knapsack_ilp(values, weights, capacity)
    # optimal: items 2+3 (100+120=220, weight 50)
    check("knapsack value = 220", abs(res["value"] - 220) < 1e-6, f"{res}")
    check("knapsack picks items 1,2 (0-indexed)", res["x"] == [0, 1, 1], f"{res['x']}")

    # ---- 6. larger random 0/1 knapsack vs brute -----------------------------------------
    rng = _lcg(7)
    ok = True
    for trial in range(10):
        n = 5 + int(rng() * 3)
        vals = [1 + int(rng() * 20) for _ in range(n)]
        wts = [1 + int(rng() * 10) for _ in range(n)]
        cap = int(sum(wts) * 0.5)
        res = solve_knapsack_ilp(vals, wts, cap)
        cons = [(list(wts), "<=", cap)]
        bval, bx = brute_ilp(vals, cons, True, [(0, 1)] * n)
        if abs(res["value"] - bval) > 1e-6:
            ok = False
            check("knapsack == brute", False, f"trial {trial}: {res['value']} vs {bval}")
            break
    if ok:
        check("0/1 knapsack == brute force (10 instances)", True)

    # ---- 7. minimize direction --------------------------------------------------------
    # minimize 2x+3y s.t. x+y>=4, x,y>=0 integer -> (4,0)=8
    c = [2, 3]
    cons = [([1, 1], ">=", 4)]
    res = solve_ilp(c, cons, maximize=False, var_bounds={j: (0, 10) for j in range(2)})
    bval, bx = brute_ilp(c, cons, False, [(0, 10)] * 2)
    check("minimize ILP == brute", abs(res["value"] - bval) < 1e-6, f"{res['value']} vs {bval}")

    # ---- 8. integer optimum never beats the LP relaxation -------------------------------
    c = [7, 2, 5]
    cons = [([3, 1, 2], "<=", 10), ([1, 2, 1], "<=", 8)]
    ilp = solve_ilp(c, cons, maximize=True, var_bounds={j: (0, 6) for j in range(3)})
    lp = lp_relaxation_bound(c, cons, maximize=True)
    check("ILP <= LP relaxation", ilp["value"] <= lp + 1e-6, f"ILP {ilp['value']} LP {lp}")

    # ---- 9. all-integer LP needs no branching (nodes small) -----------------------------
    # a problem whose LP optimum is already integer
    c = [1, 1]
    cons = [([1, 0], "<=", 3), ([0, 1], "<=", 2)]
    res = solve_ilp(c, cons, maximize=True, var_bounds={j: (0, 10) for j in range(2)})
    check("integer LP optimum found", abs(res["value"] - 5) < 1e-6 and res["x"] == [3, 2], f"{res}")

    # ---- 10. infeasible integer problem -------------------------------------------------
    # x <= 0.5 and x >= 0.5 with x integer -> no integer feasible (but LP feasible at 0.5)
    c = [1]
    cons = [([1], "<=", 0), ([1], ">=", 0)]  # forces x=0 actually; use a real infeasible-integer case
    # x in [0.4, 0.6] integer -> infeasible
    res = solve_ilp([1], [([1], ">=", 0.4), ([1], "<=", 0.6)], maximize=True,
                    var_bounds={0: (0, 5)})
    check("no integer in (0.4,0.6) -> infeasible", res["status"] == "infeasible", f"{res}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
