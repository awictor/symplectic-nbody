"""Tests for sequence acceleration: geometric exact, Leibniz->pi, ln2, unchanged if converged."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sequence_acceleration import (  # noqa: E402
    aitken,
    aitken_iterated,
    wynn_epsilon,
    euler_transform,
    partial_sums,
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


def main():
    # ---- 1. Aitken recovers a geometric sequence limit in one step ----------------------
    # s_n = 1 - r^n -> limit 1; partial sums of geometric series
    r = 0.5
    seq = [sum(r ** k for k in range(n + 1)) for n in range(6)]  # -> 1/(1-r) = 2
    acc = aitken(seq)
    check("Aitken on geometric partial sums hits the limit",
          all(abs(a - 2.0) < 1e-9 for a in acc), f"{acc}")

    # ---- 2. Leibniz series for pi/4: acceleration beats raw sum -------------------------
    def leibniz(k):
        return (-1) ** k / (2 * k + 1)
    raw = partial_sums(leibniz, 20)  # 20 terms
    raw_err = abs(raw[-1] * 4 - math.pi)
    wynn = wynn_epsilon(raw)
    wynn_err = abs(wynn * 4 - math.pi)
    check("Wynn accelerates Leibniz far past raw sum",
          wynn_err < raw_err / 1000, f"raw {raw_err:.2e}, wynn {wynn_err:.2e}")
    check("Wynn Leibniz reaches many digits of pi", wynn_err < 1e-8, f"{wynn_err:.2e}")

    # ---- 3. iterated Aitken on Leibniz --------------------------------------------------
    est = aitken_iterated(raw)
    check("iterated Aitken accelerates Leibniz", abs(est * 4 - math.pi) < 1e-4,
          f"{abs(est*4 - math.pi):.2e}")

    # ---- 4. alternating ln 2 series -----------------------------------------------------
    # sum (-1)^(k) / (k+1) = ln 2 (k from 0): 1 - 1/2 + 1/3 - ...
    def ln2_term(k):
        return (-1) ** k / (k + 1)
    raw = partial_sums(ln2_term, 20)
    wynn = wynn_epsilon(raw)
    check("Wynn accelerates ln2 series", abs(wynn - math.log(2)) < 1e-8,
          f"{abs(wynn - math.log(2)):.2e}")

    # Euler transform on the same alternating series (pass positive a_k = 1/(k+1))
    a = [1.0 / (k + 1) for k in range(25)]
    euler = euler_transform(a)
    check("Euler transform matches ln2", abs(euler[-1] - math.log(2)) < 1e-6,
          f"{abs(euler[-1] - math.log(2)):.2e}")

    # ---- 5. already-converged sequence is left unchanged --------------------------------
    seq = [1.0, 1.0, 1.0, 1.0, 1.0]
    check("Aitken leaves a constant sequence at its value",
          all(abs(a - 1.0) < 1e-9 for a in aitken(seq)))
    check("Wynn on constant sequence returns the constant", abs(wynn_epsilon(seq) - 1.0) < 1e-9)

    # ---- 6. acceleration improves accuracy monotonically for a nice series --------------
    # geometric-like: partial sums of 0.7^k -> 1/0.3
    limit = 1 / 0.3
    raw = [sum(0.7 ** k for k in range(n + 1)) for n in range(12)]
    once = aitken(raw)
    twice = aitken(once)
    err_raw = abs(raw[-1] - limit)
    err_once = abs(once[-1] - limit)
    err_twice = abs(twice[-1] - limit)
    check("each Aitken pass reduces error", err_twice < err_once < err_raw,
          f"{err_raw:.2e} -> {err_once:.2e} -> {err_twice:.2e}")

    # ---- 7. Wynn on a faster series still converges -------------------------------------
    # sum 1/k^2 = pi^2/6
    def basel(k):
        return 1.0 / (k + 1) ** 2
    raw = partial_sums(basel, 30)
    wynn = wynn_epsilon(raw)
    check("Wynn accelerates the Basel series",
          abs(wynn - math.pi ** 2 / 6) < abs(raw[-1] - math.pi ** 2 / 6),
          f"raw {abs(raw[-1]-math.pi**2/6):.2e}, wynn {abs(wynn-math.pi**2/6):.2e}")

    # ---- 8. edge cases ------------------------------------------------------------------
    check("Aitken of a too-short sequence is empty", aitken([1.0, 2.0]) == [])
    check("Wynn of a single element returns it", wynn_epsilon([5.0]) == 5.0)
    check("Wynn of empty returns None", wynn_epsilon([]) is None)
    # division-by-zero safety: a sequence that hits the limit exactly mid-way
    check("Aitken handles zero second difference", aitken([1.0, 2.0, 3.0]) == [3.0])

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
