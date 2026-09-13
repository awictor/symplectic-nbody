"""Tests for belief propagation: sum-product marginals + max-product MAP vs brute enumeration."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from belief_propagation import (  # noqa: E402
    FactorGraph,
    sum_product,
    max_product,
    brute_marginals,
    brute_map,
    brute_partition,
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


def _close(a, b, tol=1e-6):
    return all(abs(x - y) < tol for x, y in zip(a, b))


def _random_tree_fg(n_vars, rng, card=2):
    """Build a random tree factor graph: a spanning tree of variables with a pairwise factor per
    edge and a unary factor per variable."""
    variables = {f"x{i}": card for i in range(n_vars)}
    factors = []
    # unary factors
    for i in range(n_vars):
        factors.append(((f"x{i}",), [0.3 + rng() for _ in range(card)]))
    # tree edges: attach each new variable to a random earlier one
    for i in range(1, n_vars):
        j = int(rng() * i)
        table = [[0.2 + rng() for _ in range(card)] for _ in range(card)]
        factors.append(((f"x{j}", f"x{i}"), table))
    return FactorGraph(variables, factors)


def main():
    # ---- 1. sum-product marginals match brute enumeration on random trees ---------------
    rng = _lcg(2024)
    mism = 0
    tested = 0
    for _ in range(150):
        n = 2 + int(rng() * 6)
        fg = _random_tree_fg(n, rng, card=2 + int(rng() * 2))
        marg, _ = sum_product(fg)
        bmarg, _ = brute_marginals(fg)
        for v in fg.variables:
            tested += 1
            if not _close(marg[v], bmarg[v]):
                mism += 1
    check("sum-product marginals == brute (150 trees)", mism == 0, f"{mism}/{tested}")

    # ---- 2. partition function matches --------------------------------------------------
    rng = _lcg(77)
    import math
    ok = True
    for _ in range(60):
        n = 2 + int(rng() * 5)
        fg = _random_tree_fg(n, rng)
        _, logZ = sum_product(fg)
        Z = brute_partition(fg)
        if abs(math.exp(logZ) - Z) > 1e-6 * max(1, Z):
            ok = False
    check("BP logZ matches brute partition function", ok)

    # ---- 3. max-product MAP matches brute argmax ----------------------------------------
    rng = _lcg(7)
    mism = 0
    for _ in range(120):
        n = 2 + int(rng() * 5)
        fg = _random_tree_fg(n, rng)
        assignment, _ = max_product(fg)
        bmap, _ = brute_map(fg)
        # compare joint weights (ties possible) rather than raw assignment
        def weight(a):
            w = 1.0
            for fi in range(len(fg.factors)):
                w *= fg.factor_value(fi, a)
            return w
        if abs(weight(assignment) - weight(bmap)) > 1e-9:
            mism += 1
    check("max-product MAP weight == brute max weight (120 trees)", mism == 0, f"{mism}")

    # ---- 4. chain = HMM forward-backward marginals --------------------------------------
    # a 3-state chain x0-x1-x2 with unary + pairwise; compare to brute
    variables = {"x0": 3, "x1": 3, "x2": 3}
    factors = [
        (("x0",), [0.5, 0.3, 0.2]),
        (("x1",), [0.2, 0.5, 0.3]),
        (("x2",), [0.3, 0.3, 0.4]),
        (("x0", "x1"), [[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]]),
        (("x1", "x2"), [[0.6, 0.3, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]]),
    ]
    chain = FactorGraph(variables, factors)
    marg, _ = sum_product(chain)
    bmarg, _ = brute_marginals(chain)
    check("chain sum-product == brute (HMM-like)",
          all(_close(marg[v], bmarg[v]) for v in variables))

    # ---- 5. independent variables stay independent --------------------------------------
    # two variables, only unary factors -> marginals equal the normalized unaries
    variables = {"a": 2, "b": 3}
    factors = [(("a",), [3.0, 1.0]), (("b",), [1.0, 2.0, 1.0])]
    # this is a forest, not a single tree -> is_tree should be False (two components)
    fg = FactorGraph(variables, factors)
    check("disconnected graph not a tree", not fg.is_tree())

    # a connected independent-ish case: link with a uniform factor (no coupling)
    variables = {"a": 2, "b": 2}
    factors = [(("a",), [3.0, 1.0]), (("b",), [1.0, 2.0]),
               (("a", "b"), [[1.0, 1.0], [1.0, 1.0]])]  # uniform pairwise = no coupling
    fg = FactorGraph(variables, factors)
    marg, _ = sum_product(fg)
    check("uniform coupling: a marginal = normalized unary [.75,.25]",
          _close(marg["a"], [0.75, 0.25]))
    check("uniform coupling: b marginal = normalized unary [1/3,2/3]",
          _close(marg["b"], [1 / 3, 2 / 3]))

    # ---- 6. hand example: two coupled binary vars ---------------------------------------
    # factor strongly prefers a==b
    variables = {"a": 2, "b": 2}
    factors = [(("a",), [1.0, 1.0]), (("b",), [1.0, 1.0]),
               (("a", "b"), [[5.0, 1.0], [1.0, 5.0]])]
    fg = FactorGraph(variables, factors)
    marg, _ = sum_product(fg)
    check("symmetric coupling -> uniform marginals", _close(marg["a"], [0.5, 0.5]))
    mp, _ = max_product(fg)
    check("MAP has a == b", mp["a"] == mp["b"])

    # ---- 7. tree detection --------------------------------------------------------------
    # a cycle: a-b, b-c, c-a pairwise factors -> not a tree
    variables = {"a": 2, "b": 2, "c": 2}
    tab = [[1.0, 1.0], [1.0, 1.0]]
    cyc = FactorGraph(variables, [(("a", "b"), tab), (("b", "c"), tab), (("c", "a"), tab)])
    check("triangle of pairwise factors is not a tree", not cyc.is_tree())
    check("sum-product raises on non-tree", _raises(lambda: sum_product(cyc)))

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


def _raises(fn):
    try:
        fn()
        return False
    except ValueError:
        return True


if __name__ == "__main__":
    main()
