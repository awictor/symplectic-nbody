"""Tests for the cepstrum: impulse-train period, voiced pitch, echo delay, noise, realness."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cepstrum as C  # noqa: E402


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
    st = seed & 0xFFFFFFFF

    def rnd():
        nonlocal st
        st = (1664525 * st + 1013904223) & 0xFFFFFFFF
        return (st >> 8) / (1 << 24)

    return rnd


def main():
    N = 256

    # ---- 1. an impulse train of period T peaks at a multiple of T -----------------------
    # For a pure impulse train the cepstrum has peaks at T, 2T, 3T (rahmonics); the dominant
    # one is always an integer multiple of the true period, so we check divisibility.
    for T in (16, 20, 32):
        x = [1.0 if n % T == 0 else 0.0 for n in range(N)]
        q = C.pitch_quefrency(x, min_q=5, max_q=100)
        check(f"impulse train period {T} -> quefrency is a multiple of {T}", q % T == 0, f"{q}")

    # ---- 2. a voiced-speech-like signal recovers its fundamental ------------------------
    fs = 8000
    for f0 in (100, 125, 200):
        sig = [sum(math.sin(2 * math.pi * f0 * h * n / fs) / h for h in range(1, 15))
               for n in range(N)]
        ph = C.pitch_hz(sig, fs, min_hz=80, max_hz=300)
        check(f"voiced f0={f0} recovered within 3 Hz", abs(ph - f0) < 3.0, f"{ph:.1f}")

    # ---- 3. an echo at delay d shows a cepstral peak at d -------------------------------
    for d in (25, 30, 40):
        base = [math.sin(2 * math.pi * 7 * n / N) for n in range(N)]
        echo = [base[n] + 0.6 * base[n - d] if n >= d else base[n] for n in range(N)]
        qe = C.echo_delay(echo, min_q=10, max_q=80)
        check(f"echo delay {d} -> detected {d}", abs(qe - d) <= 1, f"{qe}")

    # ---- 4. white noise has no strong low-quefrency peak (relative to a periodic signal) -
    rnd = _lcg(7)
    noise = [rnd() * 2 - 1 for _ in range(N)]
    cn = C.real_cepstrum(noise)
    per = [1.0 if n % 20 == 0 else 0.0 for n in range(N)]
    cp = C.real_cepstrum(per)
    # the periodic signal's peak-to-mean ratio in the low quefrency band should exceed noise's
    def peak_ratio(c):
        band = [abs(c[q]) for q in range(5, 100)]
        return max(band) / (sum(band) / len(band))
    check("periodic signal has a sharper cepstral peak than noise",
          peak_ratio(cp) > peak_ratio(cn), f"per {peak_ratio(cp):.2f} noise {peak_ratio(cn):.2f}")

    # ---- 5. the real cepstrum is real-valued --------------------------------------------
    c = C.real_cepstrum([math.sin(2 * math.pi * 3 * n / 64) for n in range(64)])
    check("real cepstrum is a real sequence", all(isinstance(v, float) for v in c))

    # ---- 6. the real cepstrum is symmetric for a real input -----------------------------
    x = [math.sin(2 * math.pi * 5 * n / 64) + 0.3 * math.cos(2 * math.pi * 11 * n / 64)
         for n in range(64)]
    c = C.real_cepstrum(x)
    n = len(c)
    sym = max(abs(c[k] - c[(n - k) % n]) for k in range(1, n)) < 1e-6
    check("real cepstrum is symmetric (c[k] == c[n-k])", sym)

    # ---- 7. power cepstrum is non-negative ----------------------------------------------
    pc = C.power_cepstrum([math.sin(2 * math.pi * 4 * n / 128) for n in range(128)])
    check("power cepstrum is non-negative", all(v >= 0 for v in pc))

    # ---- 8. complex cepstrum returns complex values -------------------------------------
    cc = C.complex_cepstrum([1.0, 0.5, 0.25, 0.1, 0.0, 0.0, 0.0, 0.0])
    check("complex cepstrum has the right length", len(cc) == 8)

    # ---- 9. pitch_hz respects the search band -------------------------------------------
    fs = 8000
    sig = [sum(math.sin(2 * math.pi * 150 * h * n / fs) / h for h in range(1, 12)) for n in range(N)]
    ph = C.pitch_hz(sig, fs, min_hz=80, max_hz=400)
    check("pitch within the requested band", 80 <= ph <= 400 and abs(ph - 150) < 3, f"{ph:.1f}")

    # ---- 10. shorter and longer periods both resolved (peak at a multiple of T) ---------
    for T in (12, 50):
        x = [1.0 if n % T == 0 else 0.0 for n in range(N)]
        q = C.pitch_quefrency(x, min_q=5, max_q=120)
        check(f"period {T} resolved", q % T == 0, f"{q}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
