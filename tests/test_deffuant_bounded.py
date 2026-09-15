"""Validate Deffuant: mean conservation, consensus at high d, fragmentation at low d, ~1/(2d) clusters."""

import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import deffuant_bounded as db


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


def main():
    print("Deffuant bounded-confidence tests")

    # --- mean opinion conserved by an interaction ---
    ops = [0.2, 0.5, 0.8]
    m0 = db.mean_opinion(ops)
    db.interact(ops, 0, 1, d=0.5, mu=0.5)
    check(f"interaction conserves mean ({db.mean_opinion(ops):.4f} vs {m0:.4f})",
          abs(db.mean_opinion(ops) - m0) < 1e-12)

    # --- distant opinions don't interact ---
    ops = [0.1, 0.9]
    changed = db.interact(ops, 0, 1, d=0.3, mu=0.5)
    check("distant pair unchanged", not changed and ops == [0.1, 0.9])

    # --- close opinions converge toward each other ---
    ops = [0.4, 0.5]
    db.interact(ops, 0, 1, d=0.3, mu=0.5)
    check("close pair converges", abs(ops[0] - ops[1]) < 0.1 and ops[0] > 0.4 and ops[1] < 0.5)

    # --- high confidence threshold -> single consensus cluster ---
    res = db.simulate(200, d=0.5, seed=1)
    check(f"high d -> consensus (1 cluster, got {res['n_clusters']})", res["n_clusters"] == 1)

    # --- consensus value near the preserved mean (0.5 for uniform) ---
    check(f"consensus near mean 0.5 ({res['clusters'][0]:.3f})", abs(res["clusters"][0] - 0.5) < 0.05)

    # --- low confidence threshold -> multiple clusters (fragmentation) ---
    res = db.simulate(400, d=0.15, seed=1)
    check(f"low d -> fragmentation ({res['n_clusters']} clusters)", res["n_clusters"] >= 2)

    # --- cluster count grows as threshold shrinks, tracking ~1/(2d) ---
    # 1/(2d) is a rough INTERIOR estimate; the [0,1] boundaries add a small edge cluster at each end,
    # so the observed count runs a little above the formula. Check the trend and a generous band.
    counts = []
    for d in (0.5, 0.25, 0.15, 0.1):
        res = db.simulate(500, d=d, seed=2)
        counts.append(res["n_clusters"])
        pred = db.predicted_clusters(d)
        check(f"clusters near 1/(2d) at d={d} ({res['n_clusters']} vs ~{pred})",
              pred - 1 <= res["n_clusters"] <= pred + 2)
    check("cluster count grows as d shrinks", all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1)))

    # --- mean opinion preserved end to end ---
    res = db.simulate(300, d=0.3, seed=1)
    check(f"mean preserved (~0.5, got {res['mean']:.3f})", abs(res["mean"] - 0.5) < 0.05)

    # --- opinions stay in [0,1] ---
    check("opinions in [0,1]", all(0 <= x <= 1 for x in res["opinions"]))

    # --- monotone: fewer clusters as d grows ---
    c_lo = db.simulate(400, 0.1, seed=3)["n_clusters"]
    c_hi = db.simulate(400, 0.4, seed=3)["n_clusters"]
    check(f"more open-minded -> fewer clusters ({c_lo} >= {c_hi})", c_lo >= c_hi)

    # --- find_clusters on a hand case ---
    # two tight groups near 0.2 and 0.8 with d=0.3 -> 2 clusters
    clusters = db.find_clusters([0.19, 0.20, 0.21, 0.79, 0.80, 0.81], d=0.3)
    check("find_clusters splits two groups", len(clusters) == 2)
    check("cluster means correct", abs(clusters[0] - 0.20) < 1e-9 and abs(clusters[1] - 0.80) < 1e-9)

    # --- predicted clusters formula ---
    check("predicted_clusters(0.5) == 1", db.predicted_clusters(0.5) == 1)
    check("predicted_clusters(0.1) == 5", db.predicted_clusters(0.1) == 5)

    # --- deterministic ---
    a = db.simulate(100, 0.3, seed=42)
    b = db.simulate(100, 0.3, seed=42)
    check("deterministic", a["opinions"] == b["opinions"])

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
