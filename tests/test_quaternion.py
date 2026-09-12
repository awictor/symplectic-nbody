"""Tests for quaternion: rotate==matrix, round-trips, composition homomorphism, slerp properties."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quaternion import (multiply, conjugate, inverse, norm, normalize, dot,  # noqa: E402
                        from_axis_angle, to_axis_angle, rotate_vector,
                        to_matrix, from_matrix, from_euler, to_euler,
                        slerp, angle_between, IDENTITY)


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

    def axis(self):
        v = (self.u() - 0.5, self.u() - 0.5, self.u() - 0.5)
        n = math.sqrt(sum(c * c for c in v)) or 1.0
        return tuple(c / n for c in v)


def matvec(m, v):
    return tuple(sum(m[i][j] * v[j] for j in range(3)) for i in range(3))


def vclose(a, b, tol=1e-9):
    return all(abs(x - y) < tol for x, y in zip(a, b))


def same_rotation(q1, q2, tol=1e-7):
    """q and -q are the same rotation; compare up to sign."""
    q1 = normalize(q1)
    q2 = normalize(q2)
    return all(abs(a - b) < tol for a, b in zip(q1, q2)) or \
           all(abs(a + b) < tol for a, b in zip(q1, q2))


def main():
    rng = LCG(2024)

    # ---- 1. rotate_vector == matrix multiply -----------------------------------------
    mism = 0
    for _ in range(300):
        q = from_axis_angle(rng.axis(), rng.u() * 2 * math.pi)
        v = (rng.u() * 4 - 2, rng.u() * 4 - 2, rng.u() * 4 - 2)
        rv = rotate_vector(q, v)
        mv = matvec(to_matrix(q), v)
        if not vclose(rv, mv, 1e-8):
            mism += 1
    check("rotate_vector matches rotation matrix (300 cases)", mism == 0, f"{mism} mismatches")

    # ---- 2. rotations preserve length ------------------------------------------------
    lenbad = 0
    for _ in range(200):
        q = from_axis_angle(rng.axis(), rng.u() * 2 * math.pi)
        v = (rng.u() * 4 - 2, rng.u() * 4 - 2, rng.u() * 4 - 2)
        rv = rotate_vector(q, v)
        if abs(math.sqrt(sum(c * c for c in rv)) - math.sqrt(sum(c * c for c in v))) > 1e-9:
            lenbad += 1
    check("rotation preserves vector length", lenbad == 0, f"{lenbad} failures")

    # ---- 3. axis-angle round trip ----------------------------------------------------
    rt = 0
    for _ in range(200):
        axis = rng.axis()
        angle = rng.u() * math.pi          # [0, pi] so it round-trips uniquely
        q = from_axis_angle(axis, angle)
        ax2, an2 = to_axis_angle(q)
        q2 = from_axis_angle(ax2, an2)
        if not same_rotation(q, q2):
            rt += 1
    check("axis-angle round trip", rt == 0, f"{rt} failures")

    # ---- 4. matrix round trip --------------------------------------------------------
    mrt = 0
    for _ in range(200):
        q = from_axis_angle(rng.axis(), rng.u() * 2 * math.pi)
        if not same_rotation(q, from_matrix(to_matrix(q))):
            mrt += 1
    check("matrix round trip q->M->q", mrt == 0, f"{mrt} failures")

    # ---- 5. Euler round trip ---------------------------------------------------------
    ert = 0
    for _ in range(200):
        roll = rng.u() * math.pi - math.pi / 2
        pitch = rng.u() * (math.pi - 0.2) - (math.pi - 0.2) / 2   # avoid exact +/-pi/2 gimbal lock
        yaw = rng.u() * math.pi - math.pi / 2
        q = from_euler(roll, pitch, yaw)
        if not same_rotation(q, from_euler(*to_euler(q))):
            ert += 1
    check("Euler round trip q->euler->q", ert == 0, f"{ert} failures")

    # ---- 6. composition homomorphism: rotate by q1 then q2 == rotate by q2*q1 ---------
    comp = 0
    for _ in range(200):
        q1 = from_axis_angle(rng.axis(), rng.u() * 2 * math.pi)
        q2 = from_axis_angle(rng.axis(), rng.u() * 2 * math.pi)
        v = (rng.u() * 4 - 2, rng.u() * 4 - 2, rng.u() * 4 - 2)
        seq = rotate_vector(q2, rotate_vector(q1, v))
        comb = rotate_vector(multiply(q2, q1), v)
        if not vclose(seq, comb, 1e-8):
            comp += 1
    check("composition: q2 after q1 == product q2*q1", comp == 0, f"{comp} mismatches")

    # ---- 7. rotation matrix is orthonormal, det +1 -----------------------------------
    q = from_axis_angle(rng.axis(), 1.234)
    M = to_matrix(q)
    # orthonormal columns
    cols = [[M[i][j] for i in range(3)] for j in range(3)]
    ortho = all(abs(sum(cols[a][k] * cols[b][k] for k in range(3)) - (1 if a == b else 0)) < 1e-9
                for a in range(3) for b in range(3))
    det = (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
           - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
           + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
    check("rotation matrix orthonormal", ortho)
    check("rotation matrix det == +1", abs(det - 1.0) < 1e-9, f"det={det}")

    # ---- 8. known rotation: 90 deg about z sends x to y ------------------------------
    qz = from_axis_angle((0, 0, 1), math.pi / 2)
    r = rotate_vector(qz, (1, 0, 0))
    check("90deg about z: x -> y", vclose(r, (0, 1, 0), 1e-9), f"{r}")

    # ---- 9. slerp properties ----------------------------------------------------------
    q0 = from_axis_angle((0, 0, 1), 0.0)
    q1 = from_axis_angle((0, 0, 1), math.pi / 2)
    check("slerp(t=0) == q0", same_rotation(slerp(q0, q1, 0.0), q0))
    check("slerp(t=1) == q1", same_rotation(slerp(q0, q1, 1.0), q1))
    # midpoint bisects the angle
    mid = slerp(q0, q1, 0.5)
    check("slerp midpoint bisects angle",
          abs(angle_between(q0, mid) - angle_between(mid, q1)) < 1e-9
          and abs(angle_between(q0, mid) - math.pi / 4 / 2 * 2) < 0.2)
    # constant angular velocity: equal steps -> equal angle increments
    qa = from_axis_angle((1, 1, 0), 0.0)
    qb = from_axis_angle((1, 1, 0), 2.0)
    steps = [slerp(qa, qb, i / 10) for i in range(11)]
    incs = [angle_between(steps[i], steps[i + 1]) for i in range(10)]
    check("slerp has constant angular velocity",
          max(incs) - min(incs) < 1e-6, f"inc spread {max(incs)-min(incs):.2e}")

    # ---- 10. arithmetic sanity --------------------------------------------------------
    check("q * q^-1 == identity", same_rotation(multiply(q, inverse(q)), IDENTITY))
    check("unit quaternion norm 1", abs(norm(from_axis_angle((1, 2, 3), 1.0)) - 1.0) < 1e-12)
    check("identity rotates nothing", vclose(rotate_vector(IDENTITY, (3, 4, 5)), (3, 4, 5)))
    # i*j = k in Hamilton's relations: (0,1,0,0)*(0,0,1,0) = (0,0,0,1)
    check("Hamilton i*j == k", vclose(multiply((0, 1, 0, 0), (0, 0, 1, 0)), (0, 0, 0, 1)))

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
