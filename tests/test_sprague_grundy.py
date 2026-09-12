"""Tests for sprague_grundy: mex, Grundy numbers, Nim-sum, validated vs a minimax win/loss oracle."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sprague_grundy import (mex, grundy, nim_sum, nim_moves, nim_grundy,
                            subtraction_moves_factory, subtraction_grundy,
                            kayles_moves, kayles_grundy, is_losing_position)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def rand(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return self.s >> 16

    def randint(self, lo, hi):
        return lo + self.rand() % (hi - lo + 1)


# --- mex --------------------------------------------------------------------
check("mex of empty set is 0", mex([]) == 0)
check("mex {0,1,2} is 3", mex([0, 1, 2]) == 3)
check("mex {0,2,3} is 1", mex([0, 2, 3]) == 1)
check("mex {1,2,3} is 0", mex([1, 2, 3]) == 0)
check("mex ignores duplicates", mex([0, 0, 1, 1, 3]) == 2)

# --- Nim: Grundy number equals XOR of heap sizes ---------------------------
xor_ok = True
for a in range(6):
    for b in range(6):
        for c in range(6):
            heaps = (a, b, c)
            expect = a ^ b ^ c
            if nim_grundy(heaps) != expect:
                xor_ok = False
                break
check("Nim Grundy number equals XOR of heaps (all a,b,c < 6)", xor_ok)

# --- Nim-sum combinator -----------------------------------------------------
check("nim_sum XORs values", nim_sum(1, 2, 3) == 0 and nim_sum(3, 4, 5) == 2)
check("nim_sum of a single value is itself", nim_sum(7) == 7)
check("nim_sum of nothing is 0", nim_sum() == 0)

# composition: Grundy of combined game == XOR of parts (Nim heaps are independent subgames)
comp_ok = True
for a in range(1, 6):
    for b in range(1, 6):
        combined = nim_grundy((a, b))
        parts = nim_sum(nim_grundy((a,)), nim_grundy((b,)))
        if combined != parts:
            comp_ok = False
            break
check("composition: Grundy of a sum of games is the XOR of their Grundy numbers", comp_ok)

# --- subtraction game {1,2,3}: Grundy is periodic n % 4 --------------------
sub_ok = all(subtraction_grundy(n, [1, 2, 3]) == n % 4 for n in range(40))
check("subtraction game {1,2,3}: Grundy(n) == n % 4", sub_ok)

# subtraction game {1,2}: Grundy is n % 3
check("subtraction game {1,2}: Grundy(n) == n % 3",
      all(subtraction_grundy(n, [1, 2]) == n % 3 for n in range(30)))

# subtraction game {2,3}: purely periodic with period 5, block [0,0,1,1,2]
seq = [subtraction_grundy(n, [2, 3]) for n in range(20)]
check("subtraction game {2,3}: Grundy sequence is period-5 [0,0,1,1,2]",
      seq == [0, 0, 1, 1, 2] * 4)

# --- Kayles: matches the published nimber sequence -------------------------
# OEIS A002186 / standard Kayles: g(0..12) = 0,1,2,3,1,4,3,2,1,4,2,6,4
kayles_known = [0, 1, 2, 3, 1, 4, 3, 2, 1, 4, 2, 6, 4]
kg = [kayles_grundy((n,)) for n in range(13)]
check("Kayles single-row Grundy numbers match the published sequence", kg == kayles_known)

# --- THE central validation: Grundy == 0 iff a losing position -------------
# Nim (multi-heap)
nim_oracle_ok = True
rng = LCG(2026)
for _ in range(300):
    k = rng.randint(1, 4)
    heaps = tuple(sorted(rng.randint(0, 6) for _ in range(k)))
    g0 = nim_grundy(heaps) == 0
    losing = is_losing_position(tuple(sorted(x for x in heaps if x > 0)), nim_moves)
    if g0 != losing:
        nim_oracle_ok = False
        break
check("Nim: Grundy==0 iff the minimax oracle says the position is losing", nim_oracle_ok)

# subtraction game vs oracle
sub_oracle_ok = True
allowed = [1, 3, 4]
moves = subtraction_moves_factory(allowed)
for n in range(40):
    g0 = subtraction_grundy(n, allowed) == 0
    if g0 != is_losing_position(n, moves):
        sub_oracle_ok = False
        break
check("subtraction {1,3,4}: Grundy==0 iff losing (minimax oracle)", sub_oracle_ok)

# Kayles vs oracle
kayles_oracle_ok = True
for n in range(12):
    g0 = kayles_grundy((n,)) == 0
    if g0 != is_losing_position((n,), kayles_moves):
        kayles_oracle_ok = False
        break
check("Kayles: Grundy==0 iff losing (minimax oracle)", kayles_oracle_ok)

# --- terminal position has Grundy 0 ----------------------------------------
check("empty Nim position (no heaps) has Grundy 0", nim_grundy(()) == 0)
check("empty Kayles position has Grundy 0", kayles_grundy(()) == 0)

# --- winning move exists iff Grundy != 0 -----------------------------------
# from a winning Nim position, some move reaches Grundy 0; from a losing one, none does
move_ok = True
for heaps in [(1, 2, 3), (3, 4, 5), (1, 4, 5), (2, 3, 6), (5, 5, 5)]:
    ht = tuple(sorted(heaps))
    g = nim_grundy(ht)
    can_reach_zero = any(nim_grundy(p) == 0 for p in nim_moves(ht))
    if (g != 0) != can_reach_zero:
        move_ok = False
        break
check("a winning move (to Grundy 0) exists iff the position's Grundy != 0", move_ok)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all sprague_grundy tests passed")
