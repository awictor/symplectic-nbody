"""Tests for Johnson-Lindenstrauss: distance distortion within eps, unbiased, both variants."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from johnson_lindenstrauss import (  # noqa: E402
    min_dimension,
    gaussian_matrix,
    achlioptas_matrix,
    project,
    max_distortion,
    mean_sq_ratio,
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


def _random_data(n, dim, rng):
    return [[rng() * 2 - 1 for _ in range(dim)] for _ in range(n)]


def main():
    # ---- 1. distances preserved within a modest bound after projection ------------------
    rng = _lcg(2024)
    n, orig_dim = 40, 200
    data = _random_data(n, orig_dim, rng)
    # project to a generous dimension; distortion should be small
    target = 120
    M = gaussian_matrix(target, orig_dim, seed=7)
    proj = project(data, M)
    dist = max_distortion(data, proj)
    check("gaussian projection: distances preserved (distortion < 0.35)", dist < 0.35, f"{dist:.3f}")

    # ---- 2. mean squared-distance ratio is ~1 (unbiased) --------------------------------
    ratio = mean_sq_ratio(data, proj)
    check("mean squared-distance ratio ~ 1 (unbiased)", abs(ratio - 1) < 0.1, f"{ratio:.3f}")

    # ---- 3. more target dimensions -> less distortion -----------------------------------
    d_lo = max_distortion(data, project(data, gaussian_matrix(20, orig_dim, seed=3)))
    d_hi = max_distortion(data, project(data, gaussian_matrix(160, orig_dim, seed=3)))
    check("more dimensions reduce distortion", d_hi < d_lo, f"{d_lo:.3f} -> {d_hi:.3f}")

    # ---- 4. Achlioptas sparse variant also preserves distances --------------------------
    Ma = achlioptas_matrix(target, orig_dim, seed=11)
    proja = project(data, Ma)
    dista = max_distortion(data, proja)
    check("Achlioptas projection preserves distances", dista < 0.4, f"{dista:.3f}")
    check("Achlioptas mean ratio ~ 1", abs(mean_sq_ratio(data, proja) - 1) < 0.12,
          f"{mean_sq_ratio(data, proja):.3f}")
    # sparse: about 2/3 of entries are zero
    zeros = sum(1 for row in Ma for x in row if x == 0)
    frac0 = zeros / (len(Ma) * len(Ma[0]))
    check("Achlioptas matrix ~2/3 zeros", 0.6 < frac0 < 0.73, f"{frac0:.3f}")

    # ---- 5. min_dimension formula -------------------------------------------------------
    k = min_dimension(1000, 0.1)
    check("min_dimension grows like log n / eps^2", k > 0 and k == math.ceil(
        4 * math.log(1000) / (0.1 ** 2 / 2 - 0.1 ** 3 / 3)))
    check("smaller eps needs more dimensions", min_dimension(1000, 0.05) > min_dimension(1000, 0.2))
    check("more points need more dimensions", min_dimension(10000, 0.1) > min_dimension(100, 0.1))

    # ---- 6. projecting to the JL dimension meets a reasonable distortion ----------------
    # with a truly JL-sized target the theoretical guarantee is (1+/-eps); empirically we allow slack
    rng = _lcg(99)
    n2 = 30
    data2 = _random_data(n2, 300, rng)
    eps = 0.3
    k = min_dimension(n2, eps)
    M = gaussian_matrix(k, 300, seed=5)
    proj2 = project(data2, M)
    check(f"JL dimension k={k}: worst distortion within a small multiple of eps",
          max_distortion(data2, proj2) < 2 * eps, f"{max_distortion(data2, proj2):.3f}")

    # ---- 7. determinism -----------------------------------------------------------------
    m1 = gaussian_matrix(10, 20, seed=42)
    m2 = gaussian_matrix(10, 20, seed=42)
    check("deterministic under seed", m1 == m2)
    check("different seeds differ", gaussian_matrix(10, 20, seed=1) != gaussian_matrix(10, 20, seed=2))

    # ---- 8. edge cases ------------------------------------------------------------------
    try:
        min_dimension(100, 1.5)
        check("eps out of range raises", False)
    except ValueError:
        check("eps out of range raises", True)
    # identity-ish: projecting a single point works
    single = project([[1.0, 2.0, 3.0]], gaussian_matrix(2, 3, seed=1))
    check("single point projects to target dim", len(single[0]) == 2)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
