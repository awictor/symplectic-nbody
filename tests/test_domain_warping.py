"""Tests for domain warping: zero-warp reduction, bounded displacement, added structure, continuity."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import domain_warping as DW  # noqa: E402
from fbm import total_variation  # noqa: E402


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
    # ---- 1. zero warp amplitude reduces to plain fBm ------------------------------------
    d0 = DW.DomainWarp(seed=1, amplitude=0.0)
    ok = all(abs(d0.value(x * 0.1, y * 0.1, levels=1) - d0.base.fbm2(x * 0.1, y * 0.1)) < 1e-12
             for x in range(15) for y in range(15))
    check("zero warp == plain fBm", ok)
    check("levels=0 == plain fBm always",
          abs(DW.DomainWarp(seed=3, amplitude=5.0).value(2.1, 3.3, levels=0) -
              DW.DomainWarp(seed=3).base.fbm2(2.1, 3.3)) < 1e-12)

    # ---- 2. warp displacement is bounded by amplitude * fBm bound -----------------------
    d = DW.DomainWarp(seed=1, amplitude=2.0)
    bound = 2.0 * d.wx.max_amplitude()
    ok = True
    for x in range(20):
        for y in range(20):
            dx, dy = d.warp_vector(x * 0.1, y * 0.1, levels=1)
            if abs(dx) > bound + 1e-9 or abs(dy) > bound + 1e-9:
                ok = False
    check("warp displacement bounded by amplitude * fBm bound", ok, f"bound {bound:.3f}")

    # ---- 3. warping increases total variation (adds structure) --------------------------
    base_line = [d.base.fbm2(x * 0.03, 5.0) for x in range(500)]
    warp_line = [d.value(x * 0.03, 5.0, levels=1) for x in range(500)]
    check("warping increases total variation",
          total_variation(warp_line) > total_variation(base_line),
          f"base {total_variation(base_line):.2f} warped {total_variation(warp_line):.2f}")

    # ---- 4. two levels add even more structure than one ---------------------------------
    warp1 = [d.value(x * 0.03, 5.0, levels=1) for x in range(500)]
    warp2 = [d.value(x * 0.03, 5.0, levels=2) for x in range(500)]
    check("two-level warp differs from one-level",
          any(abs(warp1[i] - warp2[i]) > 1e-6 for i in range(len(warp1))))

    # ---- 5. deterministic for a fixed seed, different across seeds ----------------------
    check("same seed -> identical value",
          DW.DomainWarp(seed=5).value(1.3, 2.7, 2) == DW.DomainWarp(seed=5).value(1.3, 2.7, 2))
    check("different seed -> different value",
          DW.DomainWarp(seed=5).value(1.3, 2.7, 2) != DW.DomainWarp(seed=6).value(1.3, 2.7, 2))

    # ---- 6. continuity: nearby inputs give nearby outputs -------------------------------
    d = DW.DomainWarp(seed=1, octaves=4, amplitude=1.0)
    ok = True
    for x in range(60):
        v0 = d.value(x * 0.1, 0.5, levels=1)
        v1 = d.value(x * 0.1 + 1e-4, 0.5, levels=1)
        if abs(v1 - v0) > 5e-2:
            ok = False
    check("warped field is continuous", ok)

    # ---- 7. output stays finite and bounded across warp levels --------------------------
    m = d.base.max_amplitude()
    for levels in (0, 1, 2):
        vals = [d.value(x * 0.13, y * 0.17, levels) for x in range(20) for y in range(20)]
        check(f"levels={levels}: output within fBm amplitude bound",
              all(abs(v) <= m + 1e-9 for v in vals),
              f"max {max(abs(v) for v in vals):.3f} bound {m:.3f}")

    # ---- 8. field normalized to [0, 1] --------------------------------------------------
    fld = DW.DomainWarp(seed=1).field(20, 16, scale=0.1, levels=1)
    flat = [v for row in fld for v in row]
    check("field in [0,1]", all(0.0 <= v <= 1.0 for v in flat))
    check("field spans full range", abs(min(flat)) < 1e-9 and abs(max(flat) - 1.0) < 1e-9)
    check("field dimensions", len(fld) == 16 and len(fld[0]) == 20)

    # ---- 9. warp vector is zero when amplitude is zero ----------------------------------
    dz = DW.DomainWarp(seed=1, amplitude=0.0)
    dx, dy = dz.warp_vector(3.3, 4.4, levels=1)
    check("zero amplitude -> zero warp vector", abs(dx) < 1e-12 and abs(dy) < 1e-12)

    # ---- 10. larger amplitude -> larger displacement ------------------------------------
    small = DW.DomainWarp(seed=1, amplitude=0.5)
    large = DW.DomainWarp(seed=1, amplitude=3.0)
    ds = sum(abs(c) for c in small.warp_vector(2.0, 2.0, 1))
    dl = sum(abs(c) for c in large.warp_vector(2.0, 2.0, 1))
    check("larger amplitude -> larger warp", dl > ds, f"small {ds:.3f} large {dl:.3f}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
