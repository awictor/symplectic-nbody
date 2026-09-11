"""Tests for wave_function_collapse: rule satisfaction, determinism, contradictions, unique tilings."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wave_function_collapse import WaveFunctionCollapse, symmetric_rules, Contradiction

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


# --- coast rules: land never touches sea directly --------------------------
tiles = ["L", "C", "S"]
allowed = [("L", "L"), ("L", "C"), ("C", "L"), ("C", "C"), ("C", "S"), ("S", "C"), ("S", "S")]
rules = {d: set(allowed) for d in ["R", "L", "D", "U"]}

ok = True
for seed in range(1, 21):
    w = WaveFunctionCollapse(10, 8, tiles, rules, weights={"L": 2, "C": 1, "S": 2}, seed=seed)
    g = w.generate()
    if not w.satisfies_rules(g):
        ok = False
        break
    # explicit check: no L directly adjacent to S
    for y in range(8):
        for x in range(10):
            a = g[y][x]
            for dx, dy in [(1, 0), (0, 1)]:
                nx, ny = x + dx, y + dy
                if nx < 10 and ny < 8:
                    b = g[ny][nx]
                    if {a, b} == {"L", "S"}:
                        ok = False
check("generated grids always satisfy the adjacency rules (20 seeds)", ok)

# --- grid shape ------------------------------------------------------------
w = WaveFunctionCollapse(6, 4, tiles, rules, seed=1)
g = w.generate()
check("grid has the requested dimensions", len(g) == 4 and all(len(row) == 6 for row in g))
check("every cell is a valid tile", all(c in tiles for row in g for c in row))

# --- determinism -----------------------------------------------------------
g1 = WaveFunctionCollapse(8, 8, tiles, rules, seed=99).generate()
g2 = WaveFunctionCollapse(8, 8, tiles, rules, seed=99).generate()
check("same seed reproduces the same grid", g1 == g2)
g3 = WaveFunctionCollapse(8, 8, tiles, rules, seed=100).generate()
check("different seed generally gives a different grid", g1 != g3)

# --- a rule set forcing a unique tiling ------------------------------------
# two tiles A, B that must alternate like a checkerboard: A only allows B in every direction, B only A
checker = {d: {("A", "B"), ("B", "A")} for d in ["R", "L", "D", "U"]}
wc = WaveFunctionCollapse(6, 6, ["A", "B"], checker, seed=5)
gc = wc.generate()
check("checkerboard rules satisfied", wc.satisfies_rules(gc))
# verify it's a proper 2-coloring: color depends only on (x+y) parity
c00 = gc[0][0]
ok = True
for y in range(6):
    for x in range(6):
        expected = c00 if (x + y) % 2 == 0 else ("B" if c00 == "A" else "A")
        if gc[y][x] != expected:
            ok = False
check("checkerboard is a perfect alternating 2-coloring", ok)

# --- a single-tile rule set (self-adjacent) fills uniformly ----------------
solo = {d: {("X", "X")} for d in ["R", "L", "D", "U"]}
ws = WaveFunctionCollapse(5, 5, ["X"], solo, seed=1)
gs = ws.generate()
check("single self-adjacent tile fills the whole grid", all(c == "X" for row in gs for c in row))

# --- an unsatisfiable rule set is reported, not silently wrong -------------
# two tiles that may never be adjacent to anything (empty rules) on a >1 cell grid
empty_rules = {d: set() for d in ["R", "L", "D", "U"]}
raised = False
try:
    WaveFunctionCollapse(3, 3, ["A", "B"], empty_rules, seed=1).generate(max_restarts=5)
except Contradiction:
    raised = True
check("unsatisfiable rule set raises Contradiction", raised)

# --- a 1x1 grid always succeeds (no neighbours to constrain) ---------------
tiny = WaveFunctionCollapse(1, 1, ["A", "B"], empty_rules, seed=1)
gt = tiny.generate()
check("1x1 grid collapses to a single tile", len(gt) == 1 and len(gt[0]) == 1 and gt[0][0] in ["A", "B"])

# --- symmetric_rules builds opposite-direction pairs -----------------------
sym = symmetric_rules({"R": {("A", "B")}, "D": {("A", "A")}})
check("symmetric_rules adds the reverse pair", ("B", "A") in sym["L"] and ("A", "B") in sym["R"])
check("symmetric_rules mirrors vertical", ("A", "A") in sym["U"] and ("A", "A") in sym["D"])

# --- a pipe/road tileset produces connected-looking output that obeys rules -
# tiles: '.' empty, '-' horizontal, '|' vertical -- simplified adjacency
pipe_tiles = [".", "-", "|"]
pipe_allowed = {
    "R": {(".", "."), (".", "-"), ("-", "-"), ("-", "."), (".", "|"), ("|", "."), ("|", "|")},
    "D": {(".", "."), (".", "|"), ("|", "|"), ("|", "."), (".", "-"), ("-", "."), ("-", "-")},
}
pipe_allowed["L"] = {(b, a) for a, b in pipe_allowed["R"]}
pipe_allowed["U"] = {(b, a) for a, b in pipe_allowed["D"]}
wp = WaveFunctionCollapse(10, 10, pipe_tiles, pipe_allowed, seed=7)
gp = wp.generate()
check("pipe tileset satisfies its rules", wp.satisfies_rules(gp))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all wave_function_collapse tests passed")
