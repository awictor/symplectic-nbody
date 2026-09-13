"""Tests for Hilbert transform: cos->sin, envelope of AM, chirp instantaneous freq, H[H[x]]=-x."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hilbert import (  # noqa: E402
    hilbert_transform,
    analytic_signal,
    envelope,
    instantaneous_phase,
    instantaneous_frequency,
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


def _maxabs(a, b):
    return max(abs(a[i] - b[i]) for i in range(len(a)))


def main():
    n = 128
    # ---- 1. Hilbert transform of cos is sin ---------------------------------------------
    # x = cos(2 pi k t), H[x] = sin(2 pi k t). Use an integer number of cycles to avoid edge leakage.
    k = 4
    t = [i / n for i in range(n)]
    cos_sig = [math.cos(2 * math.pi * k * ti) for ti in t]
    sin_sig = [math.sin(2 * math.pi * k * ti) for ti in t]
    h = hilbert_transform(cos_sig)
    check("Hilbert of cos is sin", _maxabs(h, sin_sig) < 1e-6, f"{_maxabs(h, sin_sig):.2e}")

    # ---- 2. Hilbert of sin is -cos ------------------------------------------------------
    h2 = hilbert_transform(sin_sig)
    neg_cos = [-c for c in cos_sig]
    check("Hilbert of sin is -cos", _maxabs(h2, neg_cos) < 1e-6, f"{_maxabs(h2, neg_cos):.2e}")

    # ---- 3. analytic signal real part is the original -----------------------------------
    z = analytic_signal(cos_sig)
    check("analytic signal Re == input", _maxabs([v.real for v in z], cos_sig) < 1e-9)

    # ---- 4. envelope of a constant-amplitude sinusoid is constant -----------------------
    env = envelope(cos_sig)
    # interior (avoid tiny edge effects) should be ~1
    interior = env[5:-5]
    check("envelope of unit sinusoid ~ 1", all(abs(e - 1) < 0.02 for e in interior),
          f"min {min(interior):.3f} max {max(interior):.3f}")

    # ---- 5. envelope recovers an amplitude-modulated carrier ----------------------------
    # x = (1 + 0.5 cos(2 pi 2 t)) * cos(2 pi 20 t); envelope ~ 1 + 0.5 cos(2 pi 2 t)
    N = 256
    tt = [i / N for i in range(N)]
    mod = [1 + 0.5 * math.cos(2 * math.pi * 2 * x) for x in tt]
    am = [mod[i] * math.cos(2 * math.pi * 20 * tt[i]) for i in range(N)]
    env = envelope(am)
    # compare interior envelope to the modulating amplitude
    err = max(abs(env[i] - mod[i]) for i in range(20, N - 20))
    check("envelope recovers AM modulating amplitude", err < 0.05, f"max err {err:.4f}")

    # ---- 6. instantaneous frequency of a pure tone is constant --------------------------
    fs = 128.0
    N = 256
    f0 = 10.0
    tone = [math.cos(2 * math.pi * f0 * i / fs) for i in range(N)]
    ifreq = instantaneous_frequency(tone, sample_rate=fs)
    interior = ifreq[20:-20]
    check("instantaneous freq of pure tone ~ f0", all(abs(f - f0) < 0.5 for f in interior),
          f"mean {sum(interior)/len(interior):.3f} vs {f0}")

    # ---- 7. instantaneous frequency of a chirp rises linearly ---------------------------
    # x = cos(2 pi (f0 t + 0.5 r t^2)), instantaneous freq = f0 + r t
    N = 400
    fs = 100.0
    f0 = 5.0
    r = 10.0  # Hz per second
    chirp = []
    for i in range(N):
        ti = i / fs
        chirp.append(math.cos(2 * math.pi * (f0 * ti + 0.5 * r * ti * ti)))
    ifreq = instantaneous_frequency(chirp, sample_rate=fs)
    # at interior sample i, expected freq ~ f0 + r * (i/fs)
    ok = True
    for i in range(30, N - 30):
        expected = f0 + r * (i / fs)
        if abs(ifreq[i] - expected) > 1.0:
            ok = False
    check("chirp instantaneous freq rises linearly", ok)
    # rate of rise
    lo = sum(ifreq[30:60]) / 30
    hi = sum(ifreq[-60:-30]) / 30
    check("chirp freq increases over time", hi > lo + 20, f"{lo:.2f} -> {hi:.2f}")

    # ---- 8. linearity and H[H[x]] = -x --------------------------------------------------
    a = [math.cos(2 * math.pi * 3 * ti) for ti in t]
    b = [math.sin(2 * math.pi * 5 * ti) for ti in t]
    ha = hilbert_transform(a)
    hb = hilbert_transform(b)
    hsum = hilbert_transform([a[i] + b[i] for i in range(n)])
    check("Hilbert transform is linear", _maxabs(hsum, [ha[i] + hb[i] for i in range(n)]) < 1e-6)
    # H[H[x]] = -x (for signals with no DC)
    hh = hilbert_transform(ha)
    check("H[H[x]] = -x", _maxabs(hh, [-v for v in a]) < 1e-6, f"{_maxabs(hh, [-v for v in a]):.2e}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
