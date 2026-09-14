"""Validate affinity propagation: blob recovery, exemplars-are-points, preference controls k, objective."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import affinity_propagation as ap


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


class _R:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF
        self._spare = None

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)

    def normal(self):
        if self._spare is not None:
            v = self._spare
            self._spare = None
            return v
        u1 = max(self.u(), 1e-12)
        u2 = self.u()
        r = math.sqrt(-2 * math.log(u1))
        self._spare = r * math.sin(2 * math.pi * u2)
        return r * math.cos(2 * math.pi * u2)


def blobs(centers, per, spread, rng):
    pts = []
    truth = []
    for c, ctr in enumerate(centers):
        for _ in range(per):
            pts.append([ctr[d] + spread * rng.normal() for d in range(len(ctr))])
            truth.append(c)
    return pts, truth


def main():
    print("Affinity propagation tests")

    # --- three well-separated blobs: recover 3 clusters, correct assignment ---
    rng = _R(1)
    centers = [(0.0, 0.0), (10.0, 0.0), (5.0, 9.0)]
    pts, truth = blobs(centers, 12, 0.5, rng)
    res = ap.cluster_points(pts)
    check(f"recovers 3 clusters (got {res['n_clusters']})", res["n_clusters"] == 3)

    # points in the same true blob share a cluster id; different blobs differ
    cid = res["cluster_id"]
    ok = True
    per = 12
    for b in range(3):
        ids = set(cid[b * per:(b + 1) * per])
        if len(ids) != 1:
            ok = False
    # the three blob-ids are distinct
    blob_ids = [cid[b * per] for b in range(3)]
    check("each blob is one pure cluster", ok and len(set(blob_ids)) == 3)

    # --- exemplars are actual data points ---
    check("exemplars are data-point indices",
          all(0 <= e < len(pts) for e in res["exemplars"]))
    check("one exemplar per cluster", len(res["exemplars"]) == res["n_clusters"])

    # --- each point assigned to the exemplar it is most similar to ---
    S = ap.negative_sq_euclidean(pts)
    for i in range(len(pts)):
        assigned = res["labels"][i]
        best = max(res["exemplars"], key=lambda k: S[i][k])
        if abs(S[i][assigned] - S[i][best]) > 1e-9:
            check("assignment maximizes similarity to exemplar", False)
            break
    else:
        check("assignment maximizes similarity to exemplar", True)

    # --- preference controls the number of clusters ---
    # (preference = self-similarity s(k,k), which is NEGATIVE here; a LESS-negative preference
    # makes each point cheaper to elect as its own exemplar, so more clusters. Use extra damping
    # at extreme preferences -- very negative values make the message passing slow to converge.)
    S = ap.negative_sq_euclidean(pts)
    med = ap.median_offdiagonal(S)
    low = ap.affinity_propagation(S, preference=med * 5, damping=0.7, max_iter=400)     # more negative
    high = ap.affinity_propagation(S, preference=med * 0.02, damping=0.7, max_iter=400)  # near zero
    check(f"less-negative preference -> more clusters ({low['n_clusters']} <= {high['n_clusters']})",
          low["n_clusters"] <= high["n_clusters"])
    check("near-zero preference over-segments (many exemplars)", high["n_clusters"] > 3)

    # --- net-similarity objective: more exemplars never decreases it ---
    check("more clusters -> higher (less negative) net similarity",
          high["net_similarity"] >= low["net_similarity"] - 1e-6)

    # --- two clusters recovered on a clean 2-blob set ---
    rng = _R(5)
    pts2, _ = blobs([(0.0, 0.0), (8.0, 8.0)], 15, 0.4, rng)
    res2 = ap.cluster_points(pts2)
    check(f"two blobs -> 2 clusters (got {res2['n_clusters']})", res2["n_clusters"] == 2)

    # --- deterministic ---
    a = ap.cluster_points(pts)
    b = ap.cluster_points(pts)
    check("deterministic", a["exemplars"] == b["exemplars"] and a["cluster_id"] == b["cluster_id"])

    # --- median preference default matches explicit median ---
    res_def = ap.affinity_propagation(S)
    res_med = ap.affinity_propagation(S, preference=med)
    check("default preference == median off-diagonal",
          res_def["exemplars"] == res_med["exemplars"])

    # --- similarity matrix is symmetric with zero-ish diagonal handling ---
    check("similarity is negative for distinct points", all(
        S[i][j] <= 0 for i in range(len(pts)) for j in range(len(pts)) if i != j))

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
