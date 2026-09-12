"""Tests for coin_change: min coins and ways vs brute force, greedy-fail, unmakeable."""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from coin_change import min_coins, can_make, count_ways, count_sequences

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 123
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


def brute_min(coins, amount):
    """Exhaustive minimum coins via BFS over reachable amounts."""
    from collections import deque
    seen = {0}
    q = deque([(0, 0)])
    while q:
        val, count = q.popleft()
        if val == amount:
            return count
        for c in coins:
            nv = val + c
            if nv <= amount and nv not in seen:
                seen.add(nv)
                q.append((nv, count + 1))
    return None


def brute_ways(coins, amount):
    """Count combinations by exhaustive recursion (coins usable unlimited times, order-independent)."""
    coins = sorted(set(coins))

    def rec(remaining, idx):
        if remaining == 0:
            return 1
        if remaining < 0 or idx >= len(coins):
            return 0
        # use coins[idx] zero or more times, then move on
        return rec(remaining - coins[idx], idx) + rec(remaining, idx + 1)

    return rec(amount, 0)


# --- greedy-defeating denomination -----------------------------------------
c, used = min_coins([1, 3, 4], 6)
check("min coins for 6 with {1,3,4} is 2 (greedy would give 3)", c == 2)
check("the coins used sum to 6", sum(used) == 6)
c2, _ = min_coins([1, 15, 25], 30)     # greedy: 25+1+1+1+1+1 = 6; optimum 15+15 = 2
check("min coins for 30 with {1,15,25} is 2", c2 == 2)

# --- canonical US currency (greedy and DP agree) ---------------------------
c, used = min_coins([1, 5, 10, 25], 63)
check("US 63 cents is 6 coins", c == 6)
check("US coins sum to 63", sum(used) == 63)

# --- min coins matches brute force -----------------------------------------
ok = True
for _ in range(150):
    coins = sorted(set(1 + int(rng() * 9) for _ in range(1 + int(rng() * 4))))
    amount = int(rng() * 40)
    dp, used = min_coins(coins, amount)
    bf = brute_min(coins, amount)
    if dp != bf:
        ok = False
        break
    if dp is not None and sum(used) != amount:
        ok = False
        break
check("min_coins matches brute force over 150 random instances", ok)

# --- unmakeable amounts ----------------------------------------------------
check("7 cannot be made from {2, 4}", not can_make([2, 4], 7))
check("min_coins returns None for an unmakeable amount", min_coins([2, 4], 7)[0] is None)
check("0 is always makeable with 0 coins", min_coins([3, 5], 0) == (0, []))
check("any amount makeable when a coin of 1 exists", can_make([1, 5], 999))

# --- count ways (combinations) matches brute -------------------------------
ok = True
for _ in range(150):
    coins = sorted(set(1 + int(rng() * 6) for _ in range(1 + int(rng() * 3))))
    amount = int(rng() * 30)
    if count_ways(coins, amount) != brute_ways(coins, amount):
        ok = False
        break
check("count_ways matches brute-force combination count", ok)

# --- known way-counts ------------------------------------------------------
check("ways to make 5 with {1,2,5} is 4", count_ways([1, 2, 5], 5) == 4)
check("ways to make 0 is 1 (empty)", count_ways([1, 2, 5], 0) == 1)
check("ways with no valid coins is 0", count_ways([3], 5) == 0)

# --- sequences (order matters) differ from combinations --------------------
# {1,2} making 4: combinations = {1111, 112, 22} = 3; sequences count ordered = 5
check("combinations of 4 with {1,2} is 3", count_ways([1, 2], 4) == 3)
check("sequences of 4 with {1,2} is 5", count_sequences([1, 2], 4) == 5)
check("sequences >= combinations", count_sequences([1, 2, 3], 6) >= count_ways([1, 2, 3], 6))

# --- a single coin ---------------------------------------------------------
check("amount divisible by the single coin", min_coins([5], 20) == (4, [5, 5, 5, 5]))
check("amount not divisible by the single coin", min_coins([5], 22)[0] is None)

# --- reconstructed coins are valid denominations ---------------------------
ok = True
for _ in range(50):
    coins = sorted(set(1 + int(rng() * 9) for _ in range(3)))
    amount = int(rng() * 50)
    c, used = min_coins(coins, amount)
    if c is not None:
        if len(used) != c or sum(used) != amount or any(u not in coins for u in used):
            ok = False
            break
check("reconstructed coins are valid, correct count, and sum to the amount", ok)

# --- larger amount ---------------------------------------------------------
c, used = min_coins([1, 5, 10, 25, 50], 287)
check("large amount min coins is correct count", c == len(used) and sum(used) == 287)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all coin_change tests passed")
