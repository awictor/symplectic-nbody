"""Validate Vicsek: low-noise order, high-noise disorder, monotone transition, constant speed, density effect."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import vicsek_flocking as vf


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


def main():
    print("Vicsek flocking tests")

    # --- order parameter bounds: aligned -> 1, random -> ~0 ---
    check("aligned order == 1", abs(vf.order_parameter([0.5, 0.5, 0.5, 0.5]) - 1.0) < 1e-12)
    # opposite pairs cancel -> 0
    check("opposite headings order == 0", vf.order_parameter([0.0, math.pi]) < 1e-12)

    # --- low noise: the flock aligns (high order) ---
    n, box, r = 200, 6.0, 1.0
    res_lo = vf.simulate(n, box, r, eta=0.2, steps=250, seed=1)
    check(f"low noise -> high order ({res_lo['order']:.3f})", res_lo["order"] > 0.7)

    # --- high noise: disorder (low order) ---
    res_hi = vf.simulate(n, box, r, eta=5.5, steps=250, seed=1)
    check(f"high noise -> low order ({res_hi['order']:.3f})", res_hi["order"] < 0.4)

    # --- order decreases monotonically across the noise sweep ---
    sweep = vf.noise_sweep(n, box, r, [0.5, 1.5, 2.5, 3.5, 4.5, 5.5], steps=200, seed=1)
    orders = [o for _e, o in sweep]
    # allow tiny non-monotone wiggles from noise; check overall decreasing trend
    check(f"order decreases with noise ({orders[0]:.2f} -> {orders[-1]:.2f})", orders[0] > orders[-1] + 0.3)
    # first few high, last few low
    check("low-noise end ordered", orders[0] > 0.6)
    check("high-noise end disordered", orders[-1] < 0.5)

    # --- speed stays constant: displacement per step == speed ---
    rng = vf._Rng(3)
    pos, theta = vf._init(50, 5.0, rng)
    speed = 0.05
    old = list(pos)
    vf.step(pos, theta, 1.0, 0.5, speed, 5.0, rng)
    ok = True
    for i in range(50):
        dx = pos[i][0] - old[i][0]
        dy = pos[i][1] - old[i][1]
        # account for periodic wrap
        dx -= 5.0 * round(dx / 5.0)
        dy -= 5.0 * round(dy / 5.0)
        d = math.sqrt(dx * dx + dy * dy)
        if abs(d - speed) > 1e-9:
            ok = False
    check("every particle moves exactly `speed`", ok)

    # --- higher density -> more order at fixed noise ---
    o_sparse = vf.simulate(60, 8.0, 1.0, eta=2.5, steps=200, seed=1)["order"]   # low density
    o_dense = vf.simulate(300, 8.0, 1.0, eta=2.5, steps=200, seed=1)["order"]   # high density
    check(f"higher density -> more order ({o_sparse:.2f} < {o_dense:.2f})", o_dense > o_sparse)

    # --- neighbour mean angle handles the 2pi wrap (0 and 2pi average to 0, not pi) ---
    # two particles at nearly-0 and nearly-2pi should average near 0
    pos = [(0.0, 0.0), (0.1, 0.0)]
    theta = [0.1, 2 * math.pi - 0.1]
    m = vf._neighbors_mean_angle(pos, theta, 0, 1.0, 10.0)
    check(f"circular mean handles wrap ({m:.3f} ~ 0)", abs(math.atan2(math.sin(m), math.cos(m))) < 0.05)

    # --- density formula ---
    check("density = N/box^2", abs(vf.density(100, 10.0) - 1.0) < 1e-12)

    # --- order parameter in [0,1] ---
    res = vf.simulate(100, 6.0, 1.0, eta=3.0, steps=100, seed=2)
    check("order in [0,1]", 0 <= res["order"] <= 1)

    # --- positions stay in the box ---
    check("positions periodic in [0,box)", all(0 <= x < box and 0 <= y < box for x, y in res["pos"]))

    # --- deterministic ---
    a = vf.simulate(50, 5.0, 1.0, eta=1.0, steps=50, seed=42)
    b = vf.simulate(50, 5.0, 1.0, eta=1.0, steps=50, seed=42)
    check("deterministic", a["order"] == b["order"] and a["pos"] == b["pos"])

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
