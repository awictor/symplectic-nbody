"""Validate LBG: cluster codewords, monotone distortion, centroid condition, encode/decode, 1-D vs Lloyd-Max."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import linde_buzo_gray as lbg
import lloyd_max as lm


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
    print("Linde-Buzo-Gray tests")

    # --- four well-separated clusters -> 4 codewords, one per cluster ---
    rng = _R(1)
    truth = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)]
    data = []
    labels = []
    for ci, ctr in enumerate(truth):
        for _ in range(25):
            data.append([ctr[d] + 0.4 * rng.normal() for d in range(2)])
            labels.append(ci)
    res = lbg.design(data, 4)
    check("codebook has 4 codewords", res["size"] == 4)
    # each true center should have a nearby codeword
    def nearest_cw(t):
        return min(res["codebook"], key=lambda c: sum((c[d] - t[d]) ** 2 for d in range(2)))
    ok = all(sum((nearest_cw(t)[d] - t[d]) ** 2 for d in range(2)) < 0.5 for t in truth)
    check("one codeword per cluster", ok)

    # every point encodes to the codeword of its own cluster (pure clusters)
    codes = lbg.encode(data, res["codebook"])
    from collections import defaultdict
    by_cluster = defaultdict(set)
    for i, c in enumerate(codes):
        by_cluster[labels[i]].add(c)
    check("each cluster maps to one codeword", all(len(s) == 1 for s in by_cluster.values()))

    # --- distortion decreases as the codebook doubles ---
    res_t = lbg.design(data, 4, track=True)
    dists = [d for _sz, d in res_t["split_history"]]
    check("distortion falls as codebook doubles", all(dists[i] >= dists[i + 1] for i in range(len(dists) - 1)))

    # --- Lloyd distortion decreases monotonically within a refinement ---
    cb = [lbg._centroid(data, 2)]
    # split to 2 and grab the history
    cb2 = [[cb[0][d] + 0.01 for d in range(2)], [cb[0][d] - 0.01 for d in range(2)]]
    _cb, _dist, hist = lbg._lloyd(data, cb2, max_iter=50, tol=1e-10)
    check("Lloyd distortion monotone decreasing", all(hist[i] >= hist[i + 1] - 1e-9 for i in range(len(hist) - 1)))

    # --- centroid condition: each codeword is the mean of its assigned points ---
    codes = lbg.encode(data, res["codebook"])
    buckets = defaultdict(list)
    for i, c in enumerate(codes):
        buckets[c].append(data[i])
    cond_ok = True
    for k, pts in buckets.items():
        cent = lbg._centroid(pts, 2)
        if sum((cent[d] - res["codebook"][k][d]) ** 2 for d in range(2)) > 1e-6:
            cond_ok = False
    check("codewords are cell centroids", cond_ok)

    # --- encode/decode round-trips to the nearest codeword ---
    idx = lbg.encode([data[0]], res["codebook"])
    dec = lbg.decode(idx, res["codebook"])
    near = min(res["codebook"], key=lambda c: sum((c[d] - data[0][d]) ** 2 for d in range(2)))
    check("decode returns the nearest codeword", dec[0] == list(near))

    # --- more codewords -> lower distortion ---
    d2 = lbg.design(data, 2)["distortion"]
    d4 = lbg.design(data, 4)["distortion"]
    d8 = lbg.design(data, 8)["distortion"]
    check(f"distortion decreases with size ({d2:.2f} > {d4:.2f} > {d8:.2f})", d2 > d4 > d8)

    # --- 1-D LBG roughly matches a scalar Lloyd-Max quantizer on the same Gaussian samples ---
    rng = _R(7)
    samples = [[rng.normal()] for _ in range(1500)]
    vq = lbg.design(samples, 4)
    vq_levels = sorted(c[0] for c in vq["codebook"])
    flat = [s[0] for s in samples]
    lmq = lm.design_from_samples(flat, 4)
    lm_levels = sorted(lmq["levels"])
    close = all(abs(vq_levels[i] - lm_levels[i]) < 0.3 for i in range(4))
    check("1-D LBG matches scalar Lloyd-Max levels", close)

    # --- reproducible ---
    a = lbg.design(data, 4)
    b = lbg.design(data, 4)
    check("deterministic", a["codebook"] == b["codebook"])

    # --- encode indices in range ---
    codes = lbg.encode(data, res["codebook"])
    check("all codes in [0, size)", all(0 <= c < res["size"] for c in codes))

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
