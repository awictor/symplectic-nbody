"""Validate SOM: quantization-error descent, cluster contiguity, topographic preservation, decay."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import self_organizing_map as som


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


def main():
    print("SOM tests")

    # --- quantization error decreases with training ---
    rng = _R(1)
    # data: three clusters in 3-D
    centers = [(0.2, 0.2, 0.2), (0.8, 0.8, 0.2), (0.5, 0.2, 0.8)]
    data = []
    labels = []
    for ci, ctr in enumerate(centers):
        for _ in range(20):
            data.append([ctr[d] + 0.05 * rng.normal() for d in range(3)])
            labels.append(ci)

    m = som.SOM(6, 6, 3, seed=2)
    qe_before = m.quantization_error(data)
    m, hist = m.train(data, epochs=40, alpha0=0.5, track=True)
    qe_after = m.quantization_error(data)
    check(f"quantization error drops ({qe_before:.3f} -> {qe_after:.3f})", qe_after < qe_before * 0.5)
    check("QE history ends lower than it starts", hist[-1] < hist[0])

    # --- cluster contiguity: each cluster maps to a compact grid region ---
    # collect BMU grid coords per cluster; within-cluster spread should be small
    from collections import defaultdict
    coords = defaultdict(list)
    for i, x in enumerate(data):
        coords[labels[i]].append(m.map_point(x))
    # mean pairwise grid distance within a cluster vs across clusters
    def mean_spread(pts):
        if len(pts) < 2:
            return 0.0
        tot = 0.0
        cnt = 0
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                tot += math.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1])
                cnt += 1
        return tot / cnt
    within = sum(mean_spread(coords[c]) for c in coords) / len(coords)
    # across-cluster centroid distances
    cents = {}
    for c in coords:
        rs = [p[0] for p in coords[c]]
        cs = [p[1] for p in coords[c]]
        cents[c] = (sum(rs) / len(rs), sum(cs) / len(cs))
    across = 0.0
    ac = 0
    keys = list(cents)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            across += math.hypot(cents[keys[a]][0] - cents[keys[b]][0],
                                 cents[keys[a]][1] - cents[keys[b]][1])
            ac += 1
    across /= ac
    check(f"clusters map to separated grid regions (within {within:.2f} < across {across:.2f})",
          within < across)

    # --- topographic error is low (neighborhoods preserved) ---
    te = m.topographic_error(data)
    check(f"topographic error low ({te:.3f})", te < 0.25)

    # --- decay schedules are monotonically decreasing ---
    alphas, sigmas = m.decay_schedule(epochs=40, n=len(data))
    check("learning rate decays monotonically", all(alphas[i] >= alphas[i + 1] for i in range(len(alphas) - 1)))
    check("neighborhood radius decays monotonically", all(sigmas[i] >= sigmas[i + 1] for i in range(len(sigmas) - 1)))
    check("sigma shrinks substantially", sigmas[-1] < sigmas[0] * 0.5)

    # --- 1-D order recovery: points on a line map in monotone grid order ---
    line = [[t / 30.0, 0.0] for t in range(30)]
    m1 = som.SOM(1, 10, 2, seed=3)
    m1.train(line, epochs=60, alpha0=0.5)
    cols = [m1.map_point(x)[1] for x in line]
    # cols should be (weakly) monotone in t -- but the SOM's grid orientation is arbitrary, so the
    # sheet may unfold ascending OR descending. Accept whichever direction has few inversions.
    inv_asc = sum(1 for i in range(len(cols) - 1) if cols[i] > cols[i + 1])
    inv_desc = sum(1 for i in range(len(cols) - 1) if cols[i] < cols[i + 1])
    inversions = min(inv_asc, inv_desc)
    check(f"1-D data maps in monotone grid order (inversions {inversions})", inversions <= 3)

    # --- reproducible per seed ---
    a = som.SOM(4, 4, 3, seed=7)
    a.train(data, epochs=10)
    b = som.SOM(4, 4, 3, seed=7)
    b.train(data, epochs=10)
    check("reproducible per seed", a.weights == b.weights)

    # --- BMU is genuinely the closest node ---
    x = data[0]
    r, c = m.map_point(x)
    bmu_d = math.sqrt(som._dist2(m.weights[r][c], x))
    all_d = min(math.sqrt(som._dist2(m.weights[rr][cc], x))
                for rr in range(m.rows) for cc in range(m.cols))
    check("BMU is the closest node", abs(bmu_d - all_d) < 1e-12)

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
