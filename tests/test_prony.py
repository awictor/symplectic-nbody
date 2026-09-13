"""Tests for Prony's method: recovers planted frequencies/damping, reconstructs, super-resolution."""

import cmath
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prony import prony, reconstruct  # noqa: E402


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


def _closest(vals, target):
    return min(vals, key=lambda v: abs(v - target))


def main():
    # ---- 1. single real decaying exponential --------------------------------------------
    dt = 0.01
    d = 3.0  # decay rate
    N = 40
    x = [math.exp(-d * n * dt) for n in range(N)]
    model = prony(x, p=1, dt=dt)
    check("single exponential: damping ~ 3", abs(model["damping"][0] - d) < 1e-3,
          f"{model['damping'][0]:.4f}")
    check("single exponential: zero frequency", abs(model["frequencies"][0]) < 1e-6)

    # ---- 2. pure undamped tone ----------------------------------------------------------
    dt = 0.001
    f0 = 50.0
    N = 60
    x = [math.cos(2 * math.pi * f0 * n * dt) for n in range(N)]
    # a real cosine = two complex modes at +/- f0
    model = prony(x, p=2, dt=dt)
    freqs = [abs(f) for f in model["frequencies"]]
    check("pure tone: recovers frequency 50", abs(_closest(freqs, 50) - 50) < 0.5,
          f"{[round(f,2) for f in model['frequencies']]}")
    check("pure tone: near-zero damping", all(abs(d) < 1.0 for d in model["damping"]),
          f"{[round(d,3) for d in model['damping']]}")

    # ---- 3. two damped sinusoids: exact recovery ----------------------------------------
    dt = 0.005
    N = 80
    # modes: f=20Hz d=2, f=35Hz d=5, plus conjugates (real signal). Use 4 modes.
    def signal(n):
        t = n * dt
        return (math.exp(-2 * t) * math.cos(2 * math.pi * 20 * t) +
                0.7 * math.exp(-5 * t) * math.cos(2 * math.pi * 35 * t))
    x = [signal(n) for n in range(N)]
    model = prony(x, p=4, dt=dt)
    freqs = sorted(abs(f) for f in model["frequencies"])
    # should include ~20 and ~35
    check("two sinusoids: recovers 20 Hz", any(abs(f - 20) < 1 for f in freqs), f"{[round(f,1) for f in freqs]}")
    check("two sinusoids: recovers 35 Hz", any(abs(f - 35) < 1 for f in freqs), f"{[round(f,1) for f in freqs]}")

    # ---- 4. reconstruction matches the signal to high precision -------------------------
    recon = reconstruct(model, N)
    err = max(abs(recon[n] - x[n]) for n in range(N))
    check("reconstruction matches signal", err < 1e-4, f"max err {err:.2e}")

    # ---- 5. exact reconstruction of an exponential sum (2p samples) ---------------------
    dt = 1.0
    # x[n] = 2*(0.9)^n + 3*(0.5)^n  (two real modes)
    x = [2 * 0.9 ** n + 3 * 0.5 ** n for n in range(20)]
    model = prony(x, p=2, dt=dt)
    modes = sorted(m.real for m in model["modes"])
    check("exponential sum: recovers modes 0.5 and 0.9",
          abs(modes[0] - 0.5) < 1e-4 and abs(modes[1] - 0.9) < 1e-4, f"{[round(m,4) for m in modes]}")
    recon = reconstruct(model, 20)
    check("exponential sum reconstructs exactly", max(abs(recon[n] - x[n]) for n in range(20)) < 1e-6)

    # ---- 6. super-resolution: two frequencies within one FFT bin ------------------------
    # FFT bin spacing = 1/(N*dt); place two tones closer than that
    dt = 0.01
    N = 64
    df_bin = 1 / (N * dt)  # ~1.56 Hz
    f1, f2 = 10.0, 10.0 + df_bin * 0.4  # 40% of a bin apart -- unresolvable by FFT
    def two_close(n):
        t = n * dt
        return math.cos(2 * math.pi * f1 * t) + math.cos(2 * math.pi * f2 * t)
    x = [two_close(n) for n in range(N)]
    model = prony(x, p=4, dt=dt)
    freqs = sorted(set(round(abs(f), 2) for f in model["frequencies"] if abs(f) > 1))
    near_f1 = any(abs(f - f1) < 0.2 for f in freqs)
    near_f2 = any(abs(f - f2) < 0.2 for f in freqs)
    check("super-resolution: both sub-bin frequencies recovered", near_f1 and near_f2,
          f"bin {df_bin:.2f}, sep {f2-f1:.2f}, got {sorted(freqs)}")

    # ---- 7. edge cases ------------------------------------------------------------------
    try:
        prony([1.0, 2.0], p=2)  # need >= 2p = 4 samples
        check("insufficient samples raises", False)
    except ValueError:
        check("insufficient samples raises", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
