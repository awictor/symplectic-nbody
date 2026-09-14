"""Validate fuzzy c-means: center recovery, 50/50 boundary membership, sum-to-1, monotone J, vs k-means."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import fuzzy_cmeans as fcm
import kmeans


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


def nearest_center(centers, target):
    return min(centers, key=lambda c: sum((c[d] - target[d]) ** 2 for d in range(len(target))))


def main():
    print("Fuzzy c-means tests")

    # --- recover centers on well-separated blobs ---
    rng = _R(1)
    truth = [(0.0, 0.0), (10.0, 0.0), (5.0, 9.0)]
    data = []
    for ctr in truth:
        for _ in range(25):
            data.append([ctr[d] + 0.4 * rng.normal() for d in range(2)])
    res = fcm.fuzzy_cmeans(data, c=3, m=2.0, seed=2)
    ok = all(
        sum((nearest_center(res["centers"], t)[d] - t[d]) ** 2 for d in range(2)) < 0.5
        for t in truth
    )
    check("recovers 3 blob centers", ok)

    # --- memberships always sum to 1 ---
    check("memberships sum to 1", all(abs(sum(row) - 1.0) < 1e-9 for row in res["memberships"]))

    # --- deep-in-cluster points get near-crisp membership ---
    # a point at the first center's location
    mem = fcm.predict_membership(list(truth[0]), res["centers"], m=2.0)
    check("point at a center is near-crisp", max(mem) > 0.9)

    # --- a point exactly between two centers gets ~50/50 ---
    c0 = res["centers"][0]
    c1 = res["centers"][1]
    mid = [(c0[d] + c1[d]) / 2 for d in range(2)]
    mem = fcm.predict_membership(mid, res["centers"], m=2.0)
    top2 = sorted(mem, reverse=True)[:2]
    check(f"midpoint is roughly balanced ({top2[0]:.2f}/{top2[1]:.2f})",
          abs(top2[0] - top2[1]) < 0.15)

    # --- objective decreases monotonically ---
    res_t = fcm.fuzzy_cmeans(data, c=3, m=2.0, seed=2, track=True)
    hist = res_t["history"]
    check("objective decreases monotonically", all(hist[i] >= hist[i + 1] - 1e-6 for i in range(len(hist) - 1)))

    # --- partition coefficient: high for crisp, lower for fuzzy ---
    crisp = fcm.fuzzy_cmeans(data, c=3, m=1.2, seed=2)   # near-hard
    fuzzy = fcm.fuzzy_cmeans(data, c=3, m=4.0, seed=2)   # very soft
    check(f"partition coeff higher for smaller m ({crisp['partition_coefficient']:.3f} > {fuzzy['partition_coefficient']:.3f})",
          crisp["partition_coefficient"] > fuzzy["partition_coefficient"])
    check("crisp partition coeff near 1", crisp["partition_coefficient"] > 0.9)

    # --- as m -> 1, hardened labels match k-means ---
    fcm_res = fcm.fuzzy_cmeans(data, c=3, m=1.05, seed=2)
    _kc, km_labels, _ki, _kit = kmeans.kmeans(data, 3, seed=2)
    # match by comparing induced partitions: same-cluster relationships should agree
    def same_partition(l1, l2):
        n = len(l1)
        agree = 0
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += 1
                if (l1[i] == l1[j]) == (l2[i] == l2[j]):
                    agree += 1
        return agree / total
    agreement = same_partition(fcm_res["labels"], km_labels)
    check(f"m->1 agrees with k-means partition ({agreement:.3f})", agreement > 0.95)

    # --- reproducible per seed ---
    a = fcm.fuzzy_cmeans(data, c=3, seed=5)
    b = fcm.fuzzy_cmeans(data, c=3, seed=5)
    check("reproducible per seed", a["centers"] == b["centers"])

    # --- point coincident with a center handled (no divide-by-zero) ---
    mem = fcm.predict_membership(res["centers"][1], res["centers"], m=2.0)
    check("coincident point membership valid", abs(sum(mem) - 1.0) < 1e-9 and max(mem) > 0.99)

    # --- rejects m <= 1 ---
    try:
        fcm.fuzzy_cmeans(data, c=3, m=1.0)
        check("rejects m <= 1", False)
    except ValueError:
        check("rejects m <= 1", True)

    # --- two blobs, 2 clusters ---
    rng = _R(9)
    d2 = []
    for ctr in [(0.0, 0.0), (8.0, 8.0)]:
        for _ in range(20):
            d2.append([ctr[d] + 0.3 * rng.normal() for d in range(2)])
    r2 = fcm.fuzzy_cmeans(d2, c=2, seed=1)
    check("two blobs separated (partition coeff high)", r2["partition_coefficient"] > 0.85)

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
