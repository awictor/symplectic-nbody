"""Tests for de_bruijn: De Bruijn sequences, window bijection, Eulerian paths."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from de_bruijn import (de_bruijn, de_bruijn_greedy, windows, is_de_bruijn,
                       eulerian_path)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


# --- length is k**n --------------------------------------------------------
for k, n in [(2, 1), (2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (4, 2), (5, 2), (2, 8)]:
    seq = de_bruijn(k, n)
    check(f"B({k},{n}) has length {k**n}", len(seq) == k ** n)

# --- every length-n window appears exactly once (the defining property) ----
for k, n in [(2, 3), (2, 5), (3, 3), (4, 2), (2, 10)]:
    seq = de_bruijn(k, n)
    w = windows(seq, n)
    check(f"B({k},{n}) windows are all distinct", len(set(w)) == len(w))
    check(f"B({k},{n}) windows cover all {k**n} strings", len(set(w)) == k ** n)
    check(f"B({k},{n}) passes is_de_bruijn", is_de_bruijn(seq, k, n))

# --- the classic B(2,3): a cyclic sequence containing all 8 binary triples --
seq = de_bruijn(2, 3)
triples = {tuple(int(b) for b in f"{i:03b}") for i in range(8)}
check("B(2,3) covers all 8 binary triples", set(windows(seq, 3)) == triples)

# --- symbols only from the alphabet ----------------------------------------
seq = de_bruijn(3, 4)
check("B(3,4) uses only symbols 0,1,2", set(seq) <= {0, 1, 2})

# --- custom alphabet -------------------------------------------------------
seq = de_bruijn(2, 3, alphabet="AB")
check("custom alphabet B(2,3) length 8", len(seq) == 8)
check("custom alphabet uses A,B only", set(seq) <= {"A", "B"})
check("custom alphabet windows all distinct", len(set(windows(seq, 3))) == 8)

# --- n == 1 : just the alphabet --------------------------------------------
check("B(2,1) is the two symbols", sorted(de_bruijn(2, 1)) == [0, 1])
check("B(5,1) covers all 5 symbols", set(de_bruijn(5, 1)) == set(range(5)))

# --- the greedy 'Ford' construction is also a valid De Bruijn sequence -----
for k, n in [(2, 3), (2, 4), (3, 2), (3, 3), (2, 6)]:
    g = de_bruijn_greedy(k, n)
    check(f"greedy B({k},{n}) length", len(g) == k ** n)
    check(f"greedy B({k},{n}) is valid", is_de_bruijn(g, k, n))

# --- is_de_bruijn rejects bad sequences ------------------------------------
check("wrong length rejected", not is_de_bruijn([0, 0, 1], 2, 2))
check("repeated window rejected", not is_de_bruijn([0, 0, 0, 0], 2, 2))
check("valid B(2,2) accepted", is_de_bruijn([0, 0, 1, 1], 2, 2))

# --- Eulerian path / circuit finder ----------------------------------------
# a simple directed cycle 0->1->2->0 : Eulerian circuit exists
p = eulerian_path(3, [(0, 1), (1, 2), (2, 0)])
check("triangle cycle has an Eulerian circuit", p is not None and len(p) == 4)


def uses_each_edge_once(path, edges):
    if path is None:
        return False
    used = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    from collections import Counter
    return Counter(used) == Counter(edges)


check("circuit uses each edge exactly once", uses_each_edge_once(p, [(0, 1), (1, 2), (2, 0)]))

# an Eulerian PATH (not circuit): 0->1->2, endpoints unbalanced
p2 = eulerian_path(3, [(0, 1), (1, 2)])
check("open path found", p2 == [0, 1, 2])
check("open path uses each edge once", uses_each_edge_once(p2, [(0, 1), (1, 2)]))

# the classic Konigsberg-style directed graph with all balanced degrees
edges = [(0, 1), (1, 2), (2, 0), (0, 2), (2, 1), (1, 0)]
p3 = eulerian_path(3, edges)
check("multigraph with balanced degrees has Eulerian circuit", uses_each_edge_once(p3, edges))

# no Eulerian path: a vertex with out-in == 2
check("degree-violating graph has no Eulerian path",
      eulerian_path(3, [(0, 1), (0, 2)]) is None)

# disconnected edges -> no single path covering all
check("disconnected edges rejected",
      eulerian_path(4, [(0, 1), (2, 3)]) is None)

# empty graph -> empty path
check("empty edge list -> empty path", eulerian_path(0, []) == [])

# --- cross-check: the De Bruijn sequence read from the Eulerian circuit -----
# equals building the graph and confirming windows. Already covered, but also
# confirm two different (k,n) produce the exact classic lengths.
check("B(2,4) length 16", len(de_bruijn(2, 4)) == 16)
check("B(10,2) covers all 100 pairs (PIN-pad)", is_de_bruijn(de_bruijn(10, 2), 10, 2))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all de_bruijn tests passed")
