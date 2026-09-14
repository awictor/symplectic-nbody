"""Tests for fBm: single-octave reduction, amplitude bound, turbulence/ridged signs, detail, continuity."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import fbm  # noqa: E402
from perlin import Perlin  # noqa: E402


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


def main():
    # ---- 1. one octave reduces to the base Perlin noise exactly -------------------------
    f1 = fbm.FBM(seed=1, octaves=1, gain=0.5)
    p = Perlin(1)
    ok2 = all(abs(f1.fbm2(x * 0.1, y * 0.1) - p.noise2(x * 0.1, y * 0.1)) < 1e-12
              for x in range(12) for y in range(12))
    check("1-octave fBm == raw Perlin (2-D)", ok2)
    ok1 = all(abs(f1.fbm1(x * 0.1) - p.noise1(x * 0.1)) < 1e-12 for x in range(20))
    check("1-octave fBm == raw Perlin (1-D)", ok1)

    # ---- 2. max_amplitude is the geometric series sum -----------------------------------
    f = fbm.FBM(seed=1, octaves=6, gain=0.5)
    expected = sum(0.5 ** i for i in range(6))
    check("max_amplitude == geometric sum", abs(f.max_amplitude() - expected) < 1e-12,
          f"{f.max_amplitude()} vs {expected}")

    # ---- 3. output respects the amplitude bound -----------------------------------------
    m = f.max_amplitude()
    vals = [f.fbm2(x * 0.13, y * 0.17) for x in range(30) for y in range(30)]
    check("|fBm| <= max_amplitude", all(abs(v) <= m + 1e-9 for v in vals),
          f"max abs {max(abs(v) for v in vals):.4f} bound {m:.4f}")

    # ---- 4. turbulence and ridged are non-negative --------------------------------------
    check("turbulence >= 0", all(f.turbulence2(x * 0.1, y * 0.1) >= 0
                                 for x in range(25) for y in range(25)))
    check("ridged >= 0", all(f.ridged2(x * 0.1, y * 0.1) >= 0
                             for x in range(25) for y in range(25)))

    # ---- 5. adding octaves increases total variation (more detail) ----------------------
    line1 = [fbm.FBM(seed=1, octaves=1).fbm1(x * 0.05) for x in range(300)]
    line6 = [fbm.FBM(seed=1, octaves=6).fbm1(x * 0.05) for x in range(300)]
    check("more octaves -> more total variation",
          fbm.total_variation(line6) > fbm.total_variation(line1),
          f"1:{fbm.total_variation(line1):.2f} 6:{fbm.total_variation(line6):.2f}")

    # ---- 6. lower gain -> smoother (less high-frequency variance) -----------------------
    def diff_var(vals):
        d = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        mu = sum(d) / len(d)
        return sum((x - mu) ** 2 for x in d) / len(d)
    rough = [fbm.FBM(seed=1, octaves=6, gain=0.8).fbm1(x * 0.05) for x in range(300)]
    smooth = [fbm.FBM(seed=1, octaves=6, gain=0.3).fbm1(x * 0.05) for x in range(300)]
    check("lower gain gives a smoother field", diff_var(smooth) < diff_var(rough),
          f"smooth {diff_var(smooth):.4f} rough {diff_var(rough):.4f}")

    # ---- 7. reproducible for a fixed seed, different for different seeds -----------------
    a = fbm.FBM(seed=42).fbm2(1.3, 2.7)
    b = fbm.FBM(seed=42).fbm2(1.3, 2.7)
    check("same seed -> identical value", abs(a - b) < 1e-15)
    c = fbm.FBM(seed=43).fbm2(1.3, 2.7)
    check("different seed -> different value", abs(a - c) > 1e-9)

    # ---- 8. continuity: nearby inputs give nearby outputs -------------------------------
    f = fbm.FBM(seed=1, octaves=4)
    ok = True
    for x in range(50):
        v0 = f.fbm2(x * 0.1, 0.5)
        v1 = f.fbm2(x * 0.1 + 1e-4, 0.5)
        if abs(v1 - v0) > 1e-2:
            ok = False
    check("fBm is continuous (small step -> small change)", ok)

    # ---- 9. heightmap normalized to [0, 1] ----------------------------------------------
    hm = fbm.FBM(seed=1, octaves=5).heightmap(20, 16, scale=0.15, mode="fbm")
    flat = [v for row in hm for v in row]
    check("heightmap in [0,1]", all(0.0 <= v <= 1.0 for v in flat))
    check("heightmap spans the full range", abs(min(flat)) < 1e-9 and abs(max(flat) - 1.0) < 1e-9)
    check("heightmap dimensions", len(hm) == 16 and len(hm[0]) == 20)

    # ---- 10. ridged and turbulence heightmaps also normalize ----------------------------
    for mode in ("turbulence", "ridged"):
        hm2 = fbm.FBM(seed=2, octaves=5).heightmap(16, 16, scale=0.15, mode=mode)
        f2 = [v for row in hm2 for v in row]
        check(f"{mode} heightmap in [0,1]", all(0.0 <= v <= 1.0 for v in f2))

    # ---- 11. lacunarity changes the frequency content -----------------------------------
    l2 = [fbm.FBM(seed=1, octaves=4, lacunarity=2.0).fbm1(x * 0.05) for x in range(300)]
    l3 = [fbm.FBM(seed=1, octaves=4, lacunarity=3.0).fbm1(x * 0.05) for x in range(300)]
    check("higher lacunarity -> more variation", fbm.total_variation(l3) > fbm.total_variation(l2),
          f"l2 {fbm.total_variation(l2):.2f} l3 {fbm.total_variation(l3):.2f}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
