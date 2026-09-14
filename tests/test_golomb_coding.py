"""Validate Golomb: round-trip, prefix-freeness, truncated-binary minimality, near-entropy, Rice, optimal m."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import golomb_coding as gc


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


class _R:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)


def geometric_sample(p, n, seed):
    """Sample n values from P(k) = (1-p) p^k via inverse-CDF on the geometric."""
    rng = _R(seed)
    out = []
    for _ in range(n):
        u = rng.u()
        # number of failures before first success with success prob (1-p): floor(log(1-u)/log(p))
        k = int(math.log(1 - u + 1e-18) / math.log(p))
        out.append(max(0, k))
    return out


def main():
    print("Golomb coding tests")

    # --- round-trip for every m and a range of values ---
    ok = True
    for m in range(1, 12):
        for n in range(0, 40):
            bits = gc.encode_int(n, m)
            v, i = gc.decode_int(bits, m, 0)
            if v != n or i != len(bits):
                ok = False
    check("encode/decode round-trips for all m, n", ok)

    # --- stream is prefix-free: decodes without delimiters ---
    for m in (1, 3, 4, 7, 16):
        vals = [0, 5, 1, 20, 3, 8, 0, 13]
        bits = gc.encode(vals, m)
        back = gc.decode(bits, m, len(vals))
        check(f"stream round-trips (m={m})", back == vals)

    # --- truncated binary uses minimal bits and is prefix-free ---
    # for m=5: k=2, cutoff = 8-5 = 3, so r=0,1,2 -> 2 bits, r=3,4 -> 3 bits
    m = 5
    codes = [gc._truncated_binary(r, m) for r in range(m)]
    lengths = [len(c) for c in codes]
    check("truncated binary short codes first", lengths == [2, 2, 2, 3, 3])
    check("truncated binary codes distinct", len(set(codes)) == m)
    # prefix-free check
    pf = all(not a.startswith(b) for a in codes for b in codes if a != b)
    check("truncated binary prefix-free", pf)

    # --- power-of-two m: truncated binary == plain binary (Rice) ---
    m = 8
    codes = [gc._truncated_binary(r, m) for r in range(m)]
    check("Rice remainders are plain fixed-width binary", all(len(c) == 3 for c in codes))

    # --- optimal m near entropy on a geometric source, beats fixed-length ---
    p = 0.75
    m_opt = gc.optimal_m(p)
    vals = geometric_sample(p, 5000, seed=1)
    mean_len = gc.mean_code_length(vals, m_opt)
    H = gc.geometric_entropy(p)
    check(f"optimal-m mean length near entropy ({mean_len:.3f} vs H={H:.3f})", mean_len < H + 0.5)
    # fixed-length code needs ceil(log2(max+1)) bits per value
    max_v = max(vals)
    fixed_len = max(1, (max_v).bit_length())
    check(f"Golomb beats fixed-length ({mean_len:.2f} < {fixed_len})", mean_len < fixed_len)

    # --- optimal m is better than nearby m values on that source ---
    len_opt = gc.mean_code_length(vals, m_opt)
    len_lo = gc.mean_code_length(vals, max(1, m_opt // 2))
    len_hi = gc.mean_code_length(vals, m_opt * 3)
    check(f"optimal m beats m/2 and 3m ({len_opt:.3f} <= min({len_lo:.3f},{len_hi:.3f}))",
          len_opt <= len_lo + 1e-9 and len_opt <= len_hi + 1e-9)

    # --- Rice coding (m=2^k) agrees with Golomb ---
    for k in (0, 1, 2, 3):
        m = gc.rice_m(k)
        for n in (0, 1, 7, 15, 100):
            b = gc.encode_int(n, m)
            v, _ = gc.decode_int(b, m, 0)
            check(f"Rice k={k} n={n} round-trips", v == n) if n == 0 and k == 0 else None
            if v != n:
                check(f"Rice k={k} n={n} round-trips", False)
    check("Rice coding round-trips (all)", True)

    # --- larger m shortens big values, lengthens small ones ---
    small_m2 = len(gc.encode_int(0, 2))
    small_m16 = len(gc.encode_int(0, 16))
    big_m2 = len(gc.encode_int(100, 2))
    big_m16 = len(gc.encode_int(100, 16))
    check(f"larger m lengthens small values ({small_m2} < {small_m16})", small_m2 < small_m16)
    check(f"larger m shortens big values ({big_m16} < {big_m2})", big_m16 < big_m2)

    # --- optimal m grows with p (slower decay -> larger m) ---
    ms = [gc.optimal_m(pp) for pp in (0.3, 0.5, 0.7, 0.9, 0.95)]
    check(f"optimal m increases with p {ms}", all(ms[i] <= ms[i + 1] for i in range(len(ms) - 1)))

    # --- deterministic ---
    check("deterministic", gc.encode_int(42, 7) == gc.encode_int(42, 7))

    # --- rejects negatives ---
    try:
        gc.encode_int(-1, 4)
        check("rejects negative", False)
    except ValueError:
        check("rejects negative", True)

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
