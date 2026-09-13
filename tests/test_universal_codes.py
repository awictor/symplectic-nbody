"""Tests for universal codes: round-trip, prefix-free, self-delimiting streams, length formulas, Golomb."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from universal_codes import (  # noqa: E402
    gamma_encode, gamma_decode_stream, gamma_length,
    delta_encode, delta_decode_stream, delta_length,
    omega_encode, omega_decode_stream,
    golomb_encode, golomb_decode_stream, golomb_optimal_M,
    rice_encode, rice_decode_stream,
    encode_stream, is_prefix_free,
)


PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def main():
    # ---- 1. gamma round-trips every n in 1..500 -----------------------------------------
    ok = all(gamma_decode_stream(gamma_encode(n)) == [n] for n in range(1, 501))
    check("gamma round-trips 1..500", ok)

    # ---- 2. delta round-trips every n in 1..500 -----------------------------------------
    ok = all(delta_decode_stream(delta_encode(n)) == [n] for n in range(1, 501))
    check("delta round-trips 1..500", ok)

    # ---- 3. omega round-trips every n in 1..500 -----------------------------------------
    ok = all(omega_decode_stream(omega_encode(n)) == [n] for n in range(1, 501))
    check("omega round-trips 1..500", ok)

    # ---- 4. Golomb round-trips for several M --------------------------------------------
    ok = True
    for M in [1, 2, 3, 5, 7, 10, 16]:
        for n in range(0, 200):
            if golomb_decode_stream(golomb_encode(n, M), M, 1) != [n]:
                ok = False
                check("golomb round-trip", False, f"M={M} n={n}")
                break
        if not ok:
            break
    check("golomb round-trips (several M, 0..199)", ok)

    # ---- 5. Rice is the M=2^k special case of Golomb ------------------------------------
    ok = True
    for k in [0, 1, 2, 3, 4]:
        for n in range(0, 100):
            if rice_encode(n, k) != golomb_encode(n, 1 << k):
                ok = False
                break
        if not ok:
            break
    check("Rice == Golomb with M=2^k", ok)
    # rice round-trips
    check("rice round-trips", all(rice_decode_stream(rice_encode(n, 3), 3, 1) == [n] for n in range(200)))

    # ---- 6. self-delimiting: a stream of many values decodes back exactly ---------------
    rng = _lcg(1)
    vals = [1 + int(rng() * 1000) for _ in range(200)]
    for scheme in ["gamma", "delta", "omega"]:
        bits = encode_stream(vals, scheme)
        dec = {"gamma": gamma_decode_stream, "delta": delta_decode_stream,
               "omega": omega_decode_stream}[scheme](bits)
        check(f"{scheme} stream self-delimiting", dec == vals, f"len {len(dec)} vs {len(vals)}")

    # ---- 7. Golomb/Rice streams self-delimiting -----------------------------------------
    nonneg = [int(rng() * 500) for _ in range(150)]
    bits = encode_stream(nonneg, "golomb", M=7)
    check("golomb stream self-delimiting", golomb_decode_stream(bits, 7, len(nonneg)) == nonneg)
    bits = encode_stream(nonneg, "rice", k=3)
    check("rice stream self-delimiting", rice_decode_stream(bits, 3, len(nonneg)) == nonneg)

    # ---- 8. codes are prefix-free -------------------------------------------------------
    for scheme, enc in [("gamma", gamma_encode), ("delta", delta_encode), ("omega", omega_encode)]:
        codes = [enc(n) for n in range(1, 60)]
        check(f"{scheme} is prefix-free", is_prefix_free(codes))
    golomb_codes = [golomb_encode(n, 5) for n in range(60)]
    check("golomb is prefix-free", is_prefix_free(golomb_codes))

    # ---- 9. code-length formulas match actual lengths -----------------------------------
    ok = all(len(gamma_encode(n)) == gamma_length(n) for n in range(1, 1000))
    check("gamma_length formula exact", ok)
    ok = all(len(delta_encode(n)) == delta_length(n) for n in range(1, 1000))
    check("delta_length formula exact", ok)

    # ---- 10. gamma known codewords ------------------------------------------------------
    check("gamma(1) = '1'", gamma_encode(1) == "1")
    check("gamma(2) = '010'", gamma_encode(2) == "010")
    check("gamma(4) = '00100'", gamma_encode(4) == "00100")
    check("gamma(5) = '00101'", gamma_encode(5) == "00101")

    # ---- 11. on a geometric source, tuned Golomb beats fixed-width ----------------------
    # geometric mean ~ 8; fixed width to hold up to max value
    rng2 = _lcg(9)
    geo = []
    for _ in range(500):
        # inverse-transform sampling of a geometric-ish variable
        u = rng2()
        val = int(-math.log(1 - u) * 8)
        geo.append(val)
    M = golomb_optimal_M(sum(geo) / len(geo))
    golomb_bits = len(encode_stream(geo, "golomb", M=M))
    maxv = max(geo)
    fixed_width = max(1, maxv.bit_length())
    fixed_bits = fixed_width * len(geo)
    check("tuned Golomb beats fixed-width on geometric",
          golomb_bits < fixed_bits, f"golomb {golomb_bits} vs fixed {fixed_bits} (M={M})")

    # ---- 12. delta shorter than gamma for large n ---------------------------------------
    check("delta < gamma for large n", delta_length(100000) < gamma_length(100000),
          f"delta {delta_length(100000)} vs gamma {gamma_length(100000)}")

    # ---- 13. input validation -----------------------------------------------------------
    for enc in [gamma_encode, delta_encode, omega_encode]:
        try:
            enc(0)
            check(f"{enc.__name__}(0) raises", False)
        except ValueError:
            check(f"{enc.__name__}(0) raises", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
