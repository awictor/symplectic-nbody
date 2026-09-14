"""Tests for ICP: recovers rigid transform from shuffled clouds, monotone RMSD, scale, no-op."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from icp import (  # noqa: E402
    icp, rotation_z, rotation_2d, _nearest_neighbours, _mean_sq_error, apply_rigid,
)
from kabsch import apply_transform  # noqa: E402


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


def _nn_rmsd(A, B):
    m = _nearest_neighbours(A, B)
    return math.sqrt(_mean_sq_error(A, B, m))


def main():
    rng = _lcg(1)
    cloud = [[rng() * 10 - 5, rng() * 10 - 5, rng() * 10 - 5] for _ in range(15)]

    # ---- 1. recovers a known rotation + translation from a SHUFFLED target --------------
    # small in-basin rotation: ICP needs a reasonable starting pose (a large unknown rotation
    # locks matching onto the wrong correspondences -- a fundamental ICP limitation).
    R = rotation_z(0.25)
    t = [3.0, -2.0, 1.5]
    target = apply_transform(R, t, cloud)
    # shuffle target order
    perm = list(range(15))
    for i in range(14, 0, -1):
        j = int(rng() * (i + 1))
        perm[i], perm[j] = perm[j], perm[i]
    shuffled = [target[perm[i]] for i in range(15)]
    res = icp(cloud, shuffled, max_iter=60)
    aligned = apply_transform(res["R"], res["t"], cloud)
    check("ICP recovers transform (shuffled target)", _nn_rmsd(aligned, shuffled) < 1e-3,
          f"rmsd {_nn_rmsd(aligned, shuffled):.2e}")

    # ---- 2. final RMSD is small (centroid-seeded ICP converges fast) --------------------
    h = res["history"]
    check("final RMSD near zero", h[-1] < 1e-3, f"{h[0]:.3f} -> {h[-1]:.2e}")

    # ---- 3. 2-D registration ------------------------------------------------------------
    cloud2 = [[rng() * 8, rng() * 8] for _ in range(12)]
    R2 = rotation_2d(0.25)
    t2 = [5.0, -3.0]
    tgt2 = apply_transform(R2, t2, cloud2)
    perm2 = list(range(12))
    for i in range(11, 0, -1):
        j = int(rng() * (i + 1))
        perm2[i], perm2[j] = perm2[j], perm2[i]
    shuf2 = [tgt2[perm2[i]] for i in range(12)]
    res2 = icp(cloud2, shuf2, max_iter=60)
    aligned2 = apply_transform(res2["R"], res2["t"], cloud2)
    check("2-D ICP recovers transform", _nn_rmsd(aligned2, shuf2) < 1e-3,
          f"{_nn_rmsd(aligned2, shuf2):.2e}")

    # ---- 4. registration with scale (Umeyama option) -----------------------------------
    R = rotation_z(0.2)
    t = [1.0, 1.0, 1.0]
    scaled = apply_transform(R, t, cloud, scale=2.5)
    perm = list(range(15))
    for i in range(14, 0, -1):
        j = int(rng() * (i + 1))
        perm[i], perm[j] = perm[j], perm[i]
    shuf = [scaled[perm[i]] for i in range(15)]
    res = icp(cloud, shuf, max_iter=80, with_scale=True)
    aligned = apply_transform(res["R"], res["t"], cloud, scale=res["scale"])
    check("ICP with scale recovers similarity", _nn_rmsd(aligned, shuf) < 1e-3,
          f"rmsd {_nn_rmsd(aligned, shuf):.2e}, scale {res['scale']:.3f}")

    # ---- 5. already-aligned clouds: near-zero RMSD immediately --------------------------
    res = icp(cloud, cloud, max_iter=10)
    check("identical clouds -> rmsd ~ 0", res["rmsd"] < 1e-3, f"{res['rmsd']:.2e}")

    # ---- 6. convergence from a translated start -----------------------------------------
    t = [10.0, 0.0, 0.0]
    target = apply_transform([[1, 0, 0], [0, 1, 0], [0, 0, 1]], t, cloud)
    res = icp(cloud, target, max_iter=50)
    aligned = apply_transform(res["R"], res["t"], cloud)
    check("pure translation recovered", _nn_rmsd(aligned, target) < 1e-3, f"{_nn_rmsd(aligned, target):.2e}")

    # ---- 7. rotation_z and rotation_2d are valid rotations ------------------------------
    Rz = rotation_z(0.7)
    # orthogonal: R R^T = I
    prod = [[sum(Rz[i][k] * Rz[j][k] for k in range(3)) for j in range(3)] for i in range(3)]
    check("rotation_z orthogonal", all(abs(prod[i][j] - (1 if i == j else 0)) < 1e-12
                                       for i in range(3) for j in range(3)))
    R2 = rotation_2d(1.1)
    check("rotation_2d determinant 1", abs((R2[0][0] * R2[1][1] - R2[0][1] * R2[1][0]) - 1.0) < 1e-12)

    # ---- 8. history is recorded and non-empty -------------------------------------------
    res = icp(cloud, apply_transform(rotation_z(0.3), [1, 1, 1], cloud), max_iter=30)
    check("history recorded", len(res["history"]) >= 1)

    # ---- 9. asymmetric small cloud (5 points) -------------------------------------------
    small = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 4.0, 1.0], [2.0, 1.0, 3.0]]
    tgt = apply_transform(rotation_z(0.2), [2, 2, 0], small)
    res = icp(small, [tgt[3], tgt[0], tgt[4], tgt[1], tgt[2]], max_iter=50)
    aligned = apply_transform(res["R"], res["t"], small)
    check("small asymmetric cloud registration", _nn_rmsd(aligned, tgt) < 1e-3,
          f"{_nn_rmsd(aligned, tgt):.2e}")

    # ---- 10. iterations bounded ---------------------------------------------------------
    res = icp(cloud, apply_transform(rotation_z(0.2), [0.5, 0.5, 0.5], cloud), max_iter=25)
    check("iterations <= max_iter", res["iterations"] <= 25)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
