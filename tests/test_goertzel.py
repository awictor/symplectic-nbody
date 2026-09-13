"""Tests for Goertzel: matches DFT bin, tone peaks in its bin, DTMF decode round-trips."""

import cmath
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from goertzel import (  # noqa: E402
    goertzel_bin,
    goertzel_power,
    goertzel_power_hz,
    dft_bin,
    dtmf_tone,
    dtmf_decode,
    DTMF_ROWS,
    DTMF_COLS,
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
    # ---- 1. Goertzel complex bin matches the direct DFT bin -----------------------------
    rng = _lcg(2024)
    maxerr = 0.0
    for _ in range(50):
        n = 8 + int(rng() * 40)
        x = [rng() * 2 - 1 for _ in range(n)]
        for k in range(n):
            g = goertzel_bin(x, k)
            d = dft_bin(x, k)
            maxerr = max(maxerr, abs(g - d))
    check("Goertzel bin == DFT bin (all k, 50 signals)", maxerr < 1e-9, f"{maxerr:.2e}")

    # ---- 2. power matches |DFT|^2 -------------------------------------------------------
    rng = _lcg(77)
    ok = True
    for _ in range(50):
        n = 16 + int(rng() * 20)
        x = [rng() * 2 - 1 for _ in range(n)]
        for k in range(n):
            p = goertzel_power(x, k)
            d = abs(dft_bin(x, k)) ** 2
            if abs(p - d) > 1e-6 * (1 + d):
                ok = False
    check("Goertzel power == |DFT|^2", ok)

    # ---- 3. a pure sinusoid peaks in exactly its own bin --------------------------------
    n = 64
    k0 = 5
    x = [math.cos(2 * math.pi * k0 * i / n) for i in range(n)]
    powers = [goertzel_power(x, k) for k in range(n // 2)]
    peak = max(range(n // 2), key=lambda k: powers[k])
    check("pure cosine peaks in its bin k0", peak == k0, f"peaked at {peak}")
    check("off-bin power much smaller than on-bin",
          powers[k0] > 100 * powers[k0 + 2], f"{powers[k0]:.1f} vs {powers[k0+2]:.4f}")

    # ---- 4. physical-frequency power --------------------------------------------------
    fs = 8000
    n = 800
    f0 = 1000
    x = [math.sin(2 * math.pi * f0 * i / fs) for i in range(n)]
    p_on = goertzel_power_hz(x, 1000, fs)
    p_off = goertzel_power_hz(x, 1500, fs)
    check("power at true frequency >> off frequency", p_on > 100 * p_off, f"{p_on:.1f} vs {p_off:.1f}")

    # ---- 5. DTMF decode round-trips every keypad digit ----------------------------------
    fs = 8000
    dur = 0.05
    digits = "0123456789ABCD*#"
    ok = True
    for dch in digits:
        tone = dtmf_tone(dch, dur, fs)
        decoded = dtmf_decode(tone, fs)
        if decoded != dch:
            ok = False
    check("DTMF decode recovers every digit", ok)

    # ---- 6. dial a whole string ---------------------------------------------------------
    fs = 8000
    dial = "8675309"
    recovered = "".join(dtmf_decode(dtmf_tone(d, 0.05, fs), fs) for d in dial)
    check("DTMF dials 867-5309 correctly", recovered == dial, recovered)

    # ---- 7. single tone / noise decodes to nothing --------------------------------------
    fs = 8000
    n = 400
    single = [math.sin(2 * math.pi * 697 * i / fs) for i in range(n)]  # only a row tone
    check("single tone decodes to None", dtmf_decode(single, fs) is None)
    rng = _lcg(7)
    noise = [rng() * 2 - 1 for _ in range(n)]
    # noise usually has no dominant dual tone
    check("noise decodes to None", dtmf_decode(noise, fs) is None)

    # ---- 8. DTMF frequencies are the standard ones --------------------------------------
    check("4 row frequencies", DTMF_ROWS == [697, 770, 852, 941])
    check("4 col frequencies", DTMF_COLS == [1209, 1336, 1477, 1633])

    # ---- 9. edge cases ------------------------------------------------------------------
    check("k=0 bin is the sum (DC)", abs(goertzel_bin([1, 2, 3, 4], 0) - 10) < 1e-9)
    try:
        dtmf_tone("Z", 0.05, 8000)
        check("bad DTMF digit raises", False)
    except ValueError:
        check("bad DTMF digit raises", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
