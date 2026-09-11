"""Tests for pagerank: analytic reference, fixed point, symmetry, dangling nodes, teleport."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pagerank import pagerank, google_matrix_apply, top_k

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


# --- probability distribution: sums to 1, all positive ---------------------
g = {"A": ["B", "C"], "B": ["C"], "C": ["A"]}
r = pagerank(g)
check("scores sum to 1", approx(sum(r.values()), 1.0, 1e-9))
check("all scores positive", all(v > 0 for v in r.values()))

# --- matches an independent dense power-iteration reference ----------------
# reference computed separately: A->B,C ; B->C ; C->A at d=0.85
ref = {"A": 0.3878, "B": 0.2148, "C": 0.3974}
check("matches dense reference A", approx(r["A"], ref["A"], 1e-3))
check("matches dense reference B", approx(r["B"], ref["B"], 1e-3))
check("matches dense reference C", approx(r["C"], ref["C"], 1e-3))
check("C is the top-ranked node", top_k(r, 1)[0][0] == "C")

# --- result is a true fixed point of the Google-matrix map -----------------
stepped = google_matrix_apply(g, r)
check("PageRank is a fixed point", max(abs(stepped[k] - r[k]) for k in r) < 1e-8)

# --- symmetry: a directed ring gives every node equal rank -----------------
ring = {0: [1], 1: [2], 2: [3], 3: [0]}
rr = pagerank(ring)
check("ring is uniform", all(approx(v, 0.25, 1e-6) for v in rr.values()))

# --- a 2-cycle is 50/50 ----------------------------------------------------
two = pagerank({"A": ["B"], "B": ["A"]})
check("2-cycle is 50/50", approx(two["A"], 0.5, 1e-6) and approx(two["B"], 0.5, 1e-6))

# --- a hub pointed to by many is ranked highest ----------------------------
star = {i: [0] for i in range(1, 6)}    # nodes 1..5 all point to hub 0
rs = pagerank(star)
check("hub ranks highest", top_k(rs, 1)[0][0] == 0)
check("hub score above the spokes", all(rs[0] > rs[i] for i in range(1, 6)))
check("spokes are equal by symmetry", all(approx(rs[1], rs[i]) for i in range(2, 6)))

# --- dangling node (no out-links) still yields a valid distribution --------
dg = {"A": ["B", "C"], "B": ["C"]}      # C has no out-links
rd = pagerank(dg)
check("dangling graph sums to 1", approx(sum(rd.values()), 1.0, 1e-9))
check("dangling node still ranked", "C" in rd and rd["C"] > 0)
# the sink C accumulates the most rank
check("sink accumulates rank", top_k(rd, 1)[0][0] == "C")

# --- damping = 0 gives the pure teleport (uniform) distribution ------------
r0 = pagerank(g, damping=0.0)
check("d=0 is uniform teleport", all(approx(v, 1.0 / 3, 1e-6) for v in r0.values()))

# --- personalization biases the ranking toward the favored node -----------
# teleport all mass to A: A should gain relative to the uniform case
rp = pagerank(g, personalization={"A": 1.0})
check("personalization sums to 1", approx(sum(rp.values()), 1.0, 1e-9))
check("personalized toward A raises A", rp["A"] > r["A"])

# --- higher damping concentrates rank (lower teleport smoothing) -----------
r_low = pagerank(g, damping=0.5)
r_high = pagerank(g, damping=0.95)
spread_low = max(r_low.values()) - min(r_low.values())
spread_high = max(r_high.values()) - min(r_high.values())
check("higher damping spreads rank more", spread_high > spread_low)

# --- empty graph -----------------------------------------------------------
check("empty graph -> empty result", pagerank({}) == {})

# --- single self-loop node -------------------------------------------------
solo = pagerank({0: [0]})
check("single node has rank 1", approx(solo[0], 1.0, 1e-9))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all pagerank tests passed")
