"""Tests for prufer: tree<->sequence bijection, Cayley's formula, vs brute enumeration."""

import os
import sys
from itertools import product

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prufer import (tree_to_prufer, prufer_to_tree, cayley_count, random_labeled_tree,
                    degree_from_prufer, is_tree, all_labeled_trees)

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


def norm(edges):
    return sorted((min(u, v), max(u, v)) for u, v in edges)


# --- known encodings --------------------------------------------------------
star = [(0, 1), (0, 2), (0, 3), (0, 4)]
check("star centered at 0 encodes to [0,0,0]", tree_to_prufer(5, star) == [0, 0, 0])
path = [(0, 1), (1, 2), (2, 3), (3, 4)]
check("path 0-1-2-3-4 encodes to [1,2,3]", tree_to_prufer(5, path) == [1, 2, 3])
check("2-vertex tree has empty Prufer sequence", tree_to_prufer(2, [(0, 1)]) == [])
check("1-vertex tree has empty Prufer sequence", tree_to_prufer(1, []) == [])

# --- decoding known sequences ----------------------------------------------
check("decode [0,0,0] gives the star", prufer_to_tree(5, [0, 0, 0]) == norm(star))
check("decode [1,2,3] gives the path", prufer_to_tree(5, [1, 2, 3]) == norm(path))
check("decode empty (n=2) gives the single edge", prufer_to_tree(2, []) == [(0, 1)])

# --- encode/decode round-trip on random trees ------------------------------
rng = LCG(2026)
roundtrip_ok = True
for _ in range(500):
    n = rng.randint(2, 30)
    # random labeled tree via attach-to-earlier
    edges = [(rng.randint(0, v - 1), v) for v in range(1, n)]
    seq = tree_to_prufer(n, edges)
    if len(seq) != max(0, n - 2):
        roundtrip_ok = False
        break
    if prufer_to_tree(n, seq) != norm(edges):
        roundtrip_ok = False
        print(f"  roundtrip fail: n={n} edges={norm(edges)} seq={seq}")
        break
check("encode-then-decode is the identity (500 random trees)", roundtrip_ok)

# --- decode-then-encode is the identity on sequences -----------------------
rng = LCG(4242)
seq_roundtrip_ok = True
for _ in range(500):
    n = rng.randint(3, 20)
    seq = [rng.randint(0, n - 1) for _ in range(n - 2)]
    edges = prufer_to_tree(n, seq)
    if tree_to_prufer(n, edges) != seq:
        seq_roundtrip_ok = False
        print(f"  seq roundtrip fail: n={n} seq={seq} back={tree_to_prufer(n, edges)}")
        break
check("decode-then-encode is the identity on sequences (500 random sequences)", seq_roundtrip_ok)

# --- every decoded sequence is a valid tree --------------------------------
rng = LCG(777)
valid_ok = True
for _ in range(500):
    n = rng.randint(2, 25)
    seq = [rng.randint(0, n - 1) for _ in range(max(0, n - 2))]
    edges = prufer_to_tree(n, seq)
    if not is_tree(n, edges):
        valid_ok = False
        print(f"  not a tree: n={n} seq={seq} edges={edges}")
        break
check("every Prufer sequence decodes to a valid tree (500 sequences)", valid_ok)

# --- degree reading: a vertex appears (degree-1) times ---------------------
rng = LCG(555)
degree_ok = True
for _ in range(300):
    n = rng.randint(2, 20)
    edges = [(rng.randint(0, v - 1), v) for v in range(1, n)]
    seq = tree_to_prufer(n, edges)
    # actual degrees
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if degree_from_prufer(n, seq) != deg:
        degree_ok = False
        break
check("Prufer degree reading (appearances + 1) matches actual degrees", degree_ok)

# --- Cayley's formula: exactly n^(n-2) distinct trees ----------------------
cayley_ok = True
for n in range(1, 8):
    trees = set(all_labeled_trees(n))
    if len(trees) != cayley_count(n):
        cayley_ok = False
        print(f"  Cayley mismatch n={n}: got {len(trees)} want {cayley_count(n)}")
        break
    # each must actually be a tree
    for t in trees:
        if not is_tree(n, list(t)):
            cayley_ok = False
            break
    if not cayley_ok:
        break
check("Cayley's formula: n^(n-2) distinct valid labeled trees (n up to 7)", cayley_ok)

# --- the Prufer-to-tree map is a bijection: all n^(n-2) sequences give -----
# ---   distinct trees, covering every labeled tree exactly once ------------
biject_ok = True
for n in range(3, 7):
    seen = set()
    for seq in product(range(n), repeat=n - 2):
        t = frozenset(prufer_to_tree(n, list(seq)))
        if t in seen:
            biject_ok = False
            break
        seen.add(t)
    if len(seen) != cayley_count(n):
        biject_ok = False
    if not biject_ok:
        break
check("distinct Prufer sequences map to distinct trees (bijection)", biject_ok)

# --- random labeled tree generator produces valid trees --------------------
gen_ok = True
for seed in range(1, 60):
    n = 3 + seed % 15
    edges = random_labeled_tree(n, seed=seed * 7919)
    if not is_tree(n, edges):
        gen_ok = False
        break
check("random_labeled_tree always produces a valid tree", gen_ok)

# --- wrong-length sequence is rejected -------------------------------------
try:
    prufer_to_tree(5, [0, 0])       # should be length 3
    raised = False
except ValueError:
    raised = True
check("a wrong-length Prufer sequence raises ValueError", raised)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all prufer tests passed")
