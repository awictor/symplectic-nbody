"""Validate Tunstall: round-trip, distinct fixed codewords, entropy bound, tightening with k, uniform balance."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import tunstall_coding as tc


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


def sample_message(probs, n, seed):
    rng = _R(seed)
    syms = sorted(probs)
    total = sum(probs.values())
    cdf = []
    acc = 0.0
    for s in syms:
        acc += probs[s] / total
        cdf.append((acc, s))
    out = []
    for _ in range(n):
        u = rng.u()
        for c, s in cdf:
            if u <= c:
                out.append(s)
                break
        else:
            out.append(syms[-1])
    return "".join(out)


def main():
    print("Tunstall coding tests")

    probs = {"a": 0.7, "b": 0.2, "c": 0.1}
    k = 4

    # --- round-trip ---
    entries = tc.build_dictionary(probs, k)
    enc, dec = tc.assign_codewords(entries, k)
    msg = sample_message(probs, 500, seed=1)
    bits = tc.encode(msg, enc)
    back = tc.decode(bits, dec, k, length=len(msg))
    check("encode/decode round-trips", back == msg)

    # --- all codewords are distinct and exactly k bits ---
    codes = list(dec.keys())
    check("codewords distinct", len(set(codes)) == len(codes))
    check("codewords are k bits", all(len(c) == k for c in codes))
    check("dictionary size <= 2^k", len(entries) <= (1 << k))

    # --- output length is a multiple of k ---
    check("output is whole codewords", len(bits) % k == 0)

    # --- bits-per-symbol lies in [entropy, entropy+1] ---
    H = tc.source_entropy(probs)
    bps = tc.bits_per_symbol(entries, k)
    check(f"bits/symbol >= entropy ({bps:.3f} >= {H:.3f})", bps >= H - 1e-9)
    check(f"bits/symbol < entropy+1 ({bps:.3f} < {H + 1:.3f})", bps < H + 1.0)

    # --- rate tightens toward entropy as k grows ---
    rates = []
    for kk in (3, 6, 9, 12):
        e = tc.build_dictionary(probs, kk)
        rates.append(tc.bits_per_symbol(e, kk))
    check(f"rate approaches entropy as k grows ({rates[0]:.3f} -> {rates[-1]:.3f}, H={H:.3f})",
          rates[-1] < rates[0] and abs(rates[-1] - H) < abs(rates[0] - H))
    check("largest-k rate close to entropy", abs(rates[-1] - H) < 0.1)

    # --- empirical encoded rate matches the predicted bits-per-symbol ---
    long_msg = sample_message(probs, 5000, seed=3)
    e12 = tc.build_dictionary(probs, 12)
    enc12, dec12 = tc.assign_codewords(e12, 12)
    b12 = tc.encode(long_msg, enc12)
    empirical_bps = len(b12) / len(long_msg)
    check(f"empirical rate ~ predicted ({empirical_bps:.3f} vs {tc.bits_per_symbol(e12, 12):.3f})",
          abs(empirical_bps - tc.bits_per_symbol(e12, 12)) < 0.15)

    # --- uniform source: dictionary strings all the same length (balanced tree) ---
    uni = {"a": 1, "b": 1, "c": 1, "d": 1}
    e = tc.build_dictionary(uni, 4)  # 4 symbols, 2^4=16 leaves -> two levels of length-2 strings
    lengths = set(len(s) for s, _p in e)
    check(f"uniform source -> equal-length strings {lengths}", len(lengths) == 1)

    # --- most-probable-leaf expansion bounds the leaf-probability spread ---
    # The Tunstall invariant: a leaf comes from expanding a node, and only the MOST probable leaf is
    # ever expanded, so max_leaf/min_leaf < 1/p_min (each leaf prob = parent * some p_i >= parent*p_min,
    # and no leaf exceeds the smallest expanded parent). This bounds the spread by 1/p_min.
    total = sum(probs.values())
    p_min = min(probs[s] / total for s in probs)
    e = tc.build_dictionary(probs, 8)
    ps = [p for _s, p in e]
    ratio = max(ps) / min(ps)
    check(f"leaf-prob spread bounded by 1/p_min (max/min {ratio:.2f} <= {1/p_min:.2f})",
          ratio <= 1.0 / p_min + 1e-9)

    # --- every single source symbol is decodable (alphabet in dictionary) ---
    for s in probs:
        check(f"symbol '{s}' encodable", s in enc or any(st.startswith(s) for st in enc))

    # --- deterministic ---
    a = tc.build_dictionary(probs, 6)
    b = tc.build_dictionary(probs, 6)
    check("deterministic", a == b)

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
