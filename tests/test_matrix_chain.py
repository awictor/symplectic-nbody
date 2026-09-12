"""Tests for matrix_chain: DP optimum vs brute force over all parenthesizations."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from matrix_chain import min_cost, order_cost, left_to_right_cost

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 77
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


def all_parenthesizations(indices):
    """All ways to fully parenthesize a list of matrix indices (as nested tuples)."""
    if len(indices) == 1:
        return [indices[0]]
    result = []
    for split in range(1, len(indices)):
        for left in all_parenthesizations(indices[:split]):
            for right in all_parenthesizations(indices[split:]):
                result.append((left, right))
    return result


def brute_min(dims):
    n = len(dims) - 1
    best = float("inf")
    for order in all_parenthesizations(list(range(1, n + 1))):
        best = min(best, order_cost(dims, order))
    return best


# --- the classic CLRS instance --------------------------------------------
dims = [30, 35, 15, 5, 10, 20, 25]
cost, parens = min_cost(dims)
check("CLRS instance optimal cost is 15125", cost == 15125)
check("CLRS parenthesization is well-formed", parens.count("(") == parens.count(")"))

# --- DP matches brute force over random chains -----------------------------
ok = True
for _ in range(60):
    n = 2 + int(rng() * 6)              # 2..7 matrices
    dims = [1 + int(rng() * 30) for _ in range(n + 1)]
    dp_cost, _ = min_cost(dims)
    bf_cost = brute_min(dims)
    if dp_cost != bf_cost:
        ok = False
        break
check("DP matches brute force over 60 random chains", ok)

# --- the reconstructed parenthesization achieves the reported cost ---------
def parse_parens(parens):
    # parse a string like "((A1A2)A3)" into a nested tuple of ints
    pos = [0]

    def parse():
        if parens[pos[0]] == "(":
            pos[0] += 1                 # consume '('
            left = parse()
            right = parse()
            pos[0] += 1                 # consume ')'
            return (left, right)
        else:
            # read "A<number>"
            assert parens[pos[0]] == "A"
            pos[0] += 1
            num = ""
            while pos[0] < len(parens) and parens[pos[0]].isdigit():
                num += parens[pos[0]]
                pos[0] += 1
            return int(num)

    return parse()


ok = True
for _ in range(40):
    n = 2 + int(rng() * 6)
    dims = [1 + int(rng() * 20) for _ in range(n + 1)]
    cost, parens = min_cost(dims)
    order = parse_parens(parens)
    if order_cost(dims, order) != cost:
        ok = False
        break
check("reconstructed parenthesization achieves the reported cost", ok)

# --- beats the naive left-to-right order on skewed dimensions --------------
skewed = [40, 20, 30, 10, 30]
opt, _ = min_cost(skewed)
naive = left_to_right_cost(skewed)
check(f"optimal beats left-to-right on skewed dims ({opt} < {naive})", opt < naive)

# --- a single matrix or two matrices ---------------------------------------
check("single matrix has zero cost", min_cost([5, 10])[0] == 0)
check("two matrices: cost is p*q*r", min_cost([5, 10, 20])[0] == 5 * 10 * 20)

# --- three matrices: pick the better of two orders -------------------------
# A(10x100) B(100x5) C(5x50): (AB)C = 10*100*5 + 10*5*50 = 5000+2500=7500;
# A(BC) = 100*5*50 + 10*100*50 = 25000+50000 = 75000 -> (AB)C is optimal
c3, _ = min_cost([10, 100, 5, 50])
check("three-matrix optimum picks the cheaper split (7500)", c3 == 7500)

# --- symmetric dimensions: order doesn't matter ----------------------------
square = [10, 10, 10, 10]
c_sq, _ = min_cost(square)
check("all-equal dimensions: any order costs the same",
      c_sq == left_to_right_cost(square))

# --- cost is never worse than left-to-right --------------------------------
ok = True
for _ in range(30):
    n = 2 + int(rng() * 6)
    dims = [1 + int(rng() * 30) for _ in range(n + 1)]
    opt, _ = min_cost(dims)
    if opt > left_to_right_cost(dims):
        ok = False
        break
check("optimal cost is never worse than left-to-right", ok)

# --- a longer chain (n=10) runs and is optimal vs left-to-right ------------
dims10 = [5, 10, 3, 12, 5, 50, 6, 8, 20, 4, 15]
opt10, p10 = min_cost(dims10)
check("10-matrix chain: optimal <= left-to-right", opt10 <= left_to_right_cost(dims10))
check("10-matrix parenthesization is balanced parens",
      p10.count("(") == p10.count(")"))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all matrix_chain tests passed")
