"""Tests for cross-correlation: autocorr peak at 0, delay recovery, periodicity, matched filter."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cross_correlation import (  # noqa: E402
    cross_correlation, autocorrelation, correlation_direct, best_lag, estimate_delay,
    normalized_correlation, matched_filter, detect_pulse, signal_energy,
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
    # ---- 1. autocorrelation peaks at lag 0 and equals the signal energy -----------------
    x = [1.0, 2, 3, 4, 3, 2, 1]
    lags, ac = autocorrelation(x)
    peak = max(range(len(ac)), key=lambda k: ac[k])
    check("autocorr peak at lag 0", lags[peak] == 0)
    check("autocorr(0) = energy", abs(ac[peak] - signal_energy(x)) < 1e-9,
          f"{ac[peak]} vs {signal_energy(x)}")

    # ---- 2. autocorrelation is symmetric ------------------------------------------------
    mid = len(ac) // 2
    check("autocorr symmetric", all(abs(ac[mid + k] - ac[mid - k]) < 1e-9
                                    for k in range(1, mid + 1)))

    # ---- 3. cross-correlation recovers a delay ------------------------------------------
    ref = [1.0, 2, 3, 4] + [0.0] * 6
    for delay in [0, 2, 3, 5]:
        delayed = [0.0] * delay + [1.0, 2, 3, 4] + [0.0] * (6 - delay)
        est = estimate_delay(ref, delayed)
        check(f"delay {delay} recovered", est == delay, f"got {est}")

    # ---- 4. FFT and direct correlation agree --------------------------------------------
    rng = _lcg(1)
    x = [rng() - 0.5 for _ in range(20)]
    y = [rng() - 0.5 for _ in range(12)]
    lags_f, vf = cross_correlation(x, y)
    lags_d, vd = correlation_direct(x, y)
    check("FFT lags == direct lags", lags_f == lags_d)
    check("FFT values == direct values", all(abs(vf[k] - vd[k]) < 1e-8 for k in range(len(vf))),
          f"max diff {max(abs(vf[k]-vd[k]) for k in range(len(vf))):.2e}")

    # ---- 5. autocorrelation of a periodic signal peaks at multiples of the period -------
    period = 8
    n = 64
    sig = [math.sin(2 * math.pi * i / period) for i in range(n)]
    lags, ac = autocorrelation(sig)
    # find peaks at positive lags that are multiples of the period
    zero = lags.index(0)
    # lag = period should be a strong local maximum
    val_at_period = ac[zero + period]
    val_between = ac[zero + period // 2]
    check("autocorr peaks at the period", val_at_period > val_between,
          f"at period {val_at_period:.2f} vs half {val_between:.2f}")

    # ---- 6. normalized correlation of a signal with itself is 1 at lag 0 ----------------
    x = [rng() for _ in range(15)]
    lags, nc = normalized_correlation(x, x)
    peak = max(range(len(nc)), key=lambda k: nc[k])
    check("normalized autocorr = 1 at lag 0", lags[peak] == 0 and abs(nc[peak] - 1.0) < 1e-9,
          f"{nc[peak]}")
    check("normalized correlation in [-1, 1]", all(-1.0001 <= v <= 1.0001 for v in nc))

    # ---- 7. matched filter finds a pulse hidden in noise --------------------------------
    rng2 = _lcg(7)
    template = [1.0, 2.0, 3.0, 2.0, 1.0]
    n = 60
    pulse_start = 25
    signal = [0.3 * (rng2() - 0.5) for _ in range(n)]  # noise
    for j in range(len(template)):
        signal[pulse_start + j] += template[j]
    detected = detect_pulse(signal, template)
    check("matched filter locates pulse", abs(detected - pulse_start) <= 1, f"got {detected}")

    # ---- 8. matched filter beats a raw threshold ----------------------------------------
    # the raw signal's argmax may be a noise spike; the matched filter's is at the pulse
    raw_argmax = max(range(n), key=lambda i: signal[i])
    mf_lags, mf_resp = matched_filter(signal, template)
    mf_argmax = mf_lags[max(range(len(mf_resp)), key=lambda k: mf_resp[k])]
    check("matched filter more reliable than raw argmax",
          abs(mf_argmax - pulse_start) <= abs(raw_argmax - pulse_start) or abs(mf_argmax - pulse_start) <= 1,
          f"mf {mf_argmax} raw {raw_argmax} true {pulse_start}")

    # ---- 9. best_lag of identical signals is 0 ------------------------------------------
    x = [3.0, 1, 4, 1, 5, 9, 2, 6]
    check("best_lag(x, x) = 0", best_lag(x, x) == 0)

    # ---- 10. cross-correlation length is n + m - 1 --------------------------------------
    lags, v = cross_correlation([1, 2, 3], [1, 1])
    check("length n+m-1", len(v) == 3 + 2 - 1)

    # ---- 11. delta functions: correlation is a shifted delta ----------------------------
    # x = delta at 0, y = delta at 0 -> peak at lag 0
    x = [1.0, 0, 0, 0]
    y = [0.0, 0, 1.0, 0]  # delta at index 2
    # x has its impulse at 0, y at 2; y is delayed by 2 relative to x
    check("delta delay", estimate_delay(x, y) == 2, f"{estimate_delay(x, y)}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
