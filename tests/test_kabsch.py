"""Tests for kabsch: recover known transforms, proper rotation, optimality, Umeyama scale."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kabsch import (kabsch, umeyama, apply_transform, rmsd, _det, _matmul, _transpose)  # noqa: E402


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


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)


def rot3(ax, ay, az):
    """3D rotation matrix from Euler angles (for generating test transforms)."""
    import math as m
    ca, sa = m.cos(ax), m.sin(ax)
    cb, sb = m.cos(ay), m.sin(ay)
    cc, sc = m.cos(az), m.sin(az)
    Rx = [[1, 0, 0], [0, ca, -sa], [0, sa, ca]]
    Ry = [[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]]
    Rz = [[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]]
    return _matmul(_matmul(Rz, Ry), Rx)


def rot2(theta):
    import math as m
    return [[m.cos(theta), -m.sin(theta)], [m.sin(theta), m.cos(theta)]]


def is_proper_rotation(R, tol=1e-4):
    n = len(R)
    RtR = _matmul(_transpose(R), R)
    ortho = all(abs(RtR[i][j] - (1 if i == j else 0)) < tol for i in range(n) for j in range(n))
    return ortho and abs(_det(R) - 1.0) < tol


def main():
    rng = LCG(2024)

    # ---- 1. recover a known 3D rotation + translation, RMSD -> 0 ----------------------
    fails = 0
    for _ in range(200):
        pts = [[rng.u() * 10, rng.u() * 10, rng.u() * 10] for _ in range(8)]
        R_true = rot3(rng.u() * math.pi, rng.u() * math.pi, rng.u() * math.pi)
        t_true = [rng.u() * 5, rng.u() * 5, rng.u() * 5]
        target = apply_transform(R_true, t_true, pts)
        res = kabsch(pts, target)
        if res["rmsd"] > 1e-3:
            fails += 1
    check("3D: recovers transform, RMSD ~ 0 (200 cases)", fails == 0, f"{fails} failures")

    # ---- 2. recovered R equals the true R --------------------------------------------
    pts = [[rng.u() * 10, rng.u() * 10, rng.u() * 10] for _ in range(10)]
    R_true = rot3(0.5, 1.1, -0.7)
    t_true = [3.0, -2.0, 1.5]
    target = apply_transform(R_true, t_true, pts)
    res = kabsch(pts, target)
    Rdiff = max(abs(res["R"][i][j] - R_true[i][j]) for i in range(3) for j in range(3))
    tdiff = max(abs(res["t"][i] - t_true[i]) for i in range(3))
    check("recovered R matches true R", Rdiff < 1e-4, f"max diff {Rdiff:.2e}")
    check("recovered t matches true t", tdiff < 1e-4, f"max diff {tdiff:.2e}")

    # ---- 3. R is always a proper rotation (never a reflection) ------------------------
    bad = 0
    for _ in range(200):
        pts = [[rng.u() * 10, rng.u() * 10, rng.u() * 10] for _ in range(6)]
        target = [[rng.u() * 10, rng.u() * 10, rng.u() * 10] for _ in range(6)]
        res = kabsch(pts, target)
        if not is_proper_rotation(res["R"]):
            bad += 1
    check("R is always a proper rotation (det +1, orthonormal)", bad == 0, f"{bad} reflections")

    # a configuration engineered to tempt a reflection (mirror-image target)
    pts = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
    mirror = [[-p[0], p[1], p[2]] for p in pts]      # reflected across x
    res = kabsch(pts, mirror)
    check("mirror target still yields a proper rotation", is_proper_rotation(res["R"]))

    # ---- 4. 2D recovery ---------------------------------------------------------------
    fails2 = 0
    for _ in range(100):
        pts = [[rng.u() * 10, rng.u() * 10] for _ in range(6)]
        R_true = rot2(rng.u() * 2 * math.pi)
        t_true = [rng.u() * 4, rng.u() * 4]
        target = apply_transform(R_true, t_true, pts)
        if kabsch(pts, target)["rmsd"] > 1e-4:
            fails2 += 1
    check("2D: recovers transform (100 cases)", fails2 == 0, f"{fails2} failures")

    # ---- 5. optimality: Kabsch RMSD <= random rotations' RMSD -------------------------
    pts = [[rng.u() * 10, rng.u() * 10, rng.u() * 10] for _ in range(12)]
    target = [[rng.u() * 10, rng.u() * 10, rng.u() * 10] for _ in range(12)]
    res = kabsch(pts, target)
    best = res["rmsd"]
    worse = 0
    for _ in range(50):
        R = rot3(rng.u() * math.pi, rng.u() * math.pi, rng.u() * math.pi)
        # translation that centers: t = centroid_target - R centroid_pts
        import statistics
        cs = [statistics.mean(p[d] for p in pts) for d in range(3)]
        ct = [statistics.mean(p[d] for p in target) for d in range(3)]
        t = [ct[i] - sum(R[i][j] * cs[j] for j in range(3)) for i in range(3)]
        r = rmsd(apply_transform(R, t, pts), target)
        if r < best - 1e-9:
            worse += 1
    check("Kabsch RMSD is optimal (no random rotation beats it)", worse == 0,
          f"{worse} random rotations were better")

    # ---- 6. noise raises RMSD gracefully, R stays close -------------------------------
    pts = [[rng.u() * 10, rng.u() * 10, rng.u() * 10] for _ in range(30)]
    R_true = rot3(0.3, -0.6, 0.9)
    t_true = [1.0, 2.0, 3.0]
    clean = apply_transform(R_true, t_true, pts)
    noisy = [[c[d] + 0.1 * (rng.u() - 0.5) for d in range(3)] for c in clean]
    res = kabsch(pts, noisy)
    Rdiff = max(abs(res["R"][i][j] - R_true[i][j]) for i in range(3) for j in range(3))
    check("with small noise RMSD is small but nonzero", 0 < res["rmsd"] < 0.2, f"{res['rmsd']:.4f}")
    check("with noise recovered R stays close to true", Rdiff < 0.05, f"max diff {Rdiff:.4f}")

    # ---- 7. Umeyama recovers a known scale --------------------------------------------
    pts = [[rng.u() * 10, rng.u() * 10, rng.u() * 10] for _ in range(10)]
    R_true = rot3(0.4, 0.8, -0.5)
    s_true = 2.5
    t_true = [1.0, -1.0, 2.0]
    target = apply_transform(R_true, t_true, pts, scale=s_true)
    res = umeyama(pts, target, with_scale=True)
    check("Umeyama recovers scale", abs(res["scale"] - s_true) < 1e-4, f"scale={res['scale']:.4f}")
    check("Umeyama RMSD ~ 0 on scaled cloud", res["rmsd"] < 1e-4, f"{res['rmsd']:.2e}")
    check("Umeyama R is proper rotation", is_proper_rotation(res["R"]))

    # without scale, Umeyama == Kabsch
    res_ns = umeyama(pts, apply_transform(R_true, t_true, pts), with_scale=False)
    check("Umeyama no-scale matches Kabsch", res_ns["rmsd"] < 1e-4 and abs(res_ns["scale"] - 1.0) < 1e-6)

    # ---- 8. degenerate inputs ---------------------------------------------------------
    check("single point aligns to zero RMSD", kabsch([[1, 2, 3]], [[4, 5, 6]])["rmsd"] < 1e-6)
    check("identical clouds -> identity-ish, zero RMSD",
          kabsch([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])["rmsd"] < 1e-6)
    try:
        kabsch([[1, 2]], [[1, 2], [3, 4]])
        check("mismatched lengths raise", False)
    except ValueError:
        check("mismatched lengths raise", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
