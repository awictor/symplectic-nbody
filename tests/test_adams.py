"""Tests for Adams methods: exact ODE recovery, convergence order ~4, PECE beats AB, invariants."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adams import (  # noqa: E402
    rk4,
    adams_bashforth,
    predictor_corrector,
    max_error,
    convergence_order,
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
    # ---- 1. exponential decay y' = -y, y(0)=1 -> exp(-t) --------------------------------
    f_decay = lambda t, y: [-y[0]]
    exact_decay = lambda t: [math.exp(-t)]
    ts, ys = predictor_corrector(f_decay, [1.0], 0, 5, 200, order=4)
    check("PECE solves exponential decay", max_error(ys, ts, exact_decay) < 1e-6,
          f"{max_error(ys, ts, exact_decay):.2e}")

    # ---- 2. harmonic oscillator y'' = -y  (y0=1, v0=0 -> cos t) -------------------------
    f_ho = lambda t, y: [y[1], -y[0]]
    exact_ho = lambda t: [math.cos(t), -math.sin(t)]
    ts, ys = predictor_corrector(f_ho, [1.0, 0.0], 0, 10, 500, order=4)
    check("PECE solves harmonic oscillator", max_error(ys, ts, exact_ho) < 1e-5,
          f"{max_error(ys, ts, exact_ho):.2e}")

    # ---- 3. logistic curve y' = y(1-y), y0=0.1 -----------------------------------------
    f_log = lambda t, y: [y[0] * (1 - y[0])]
    def exact_log(t):
        y0 = 0.1
        return [y0 * math.exp(t) / (1 - y0 + y0 * math.exp(t))]
    ts, ys = predictor_corrector(f_log, [0.1], 0, 6, 300, order=4)
    check("PECE solves logistic ODE", max_error(ys, ts, exact_log) < 1e-6,
          f"{max_error(ys, ts, exact_log):.2e}")

    # ---- 4. convergence order ~ 4 for the 4th-order PECE --------------------------------
    order = convergence_order(f_decay, [1.0], 0, 3, exact_decay, method=predictor_corrector,
                              n_coarse=20, order=4)
    check("PECE convergence order ~ 4", 3.5 < order < 4.6, f"{order:.2f}")

    # ---- 5. Adams-Bashforth convergence order matches its nominal order -----------------
    o2 = convergence_order(f_decay, [1.0], 0, 3, exact_decay, method=adams_bashforth,
                           n_coarse=40, order=2)
    o3 = convergence_order(f_decay, [1.0], 0, 3, exact_decay, method=adams_bashforth,
                           n_coarse=40, order=3)
    check("AB order-2 convergence ~ 2", 1.6 < o2 < 2.5, f"{o2:.2f}")
    check("AB order-3 convergence ~ 3", 2.5 < o3 < 3.6, f"{o3:.2f}")

    # ---- 6. predictor-corrector beats plain Adams-Bashforth at same order ---------------
    ts_ab, ys_ab = adams_bashforth(f_ho, [1.0, 0.0], 0, 10, 300, order=4)
    ts_pc, ys_pc = predictor_corrector(f_ho, [1.0, 0.0], 0, 10, 300, order=4)
    e_ab = max_error(ys_ab, ts_ab, exact_ho)
    e_pc = max_error(ys_pc, ts_pc, exact_ho)
    check("PECE more accurate than plain AB4", e_pc < e_ab, f"AB {e_ab:.2e} vs PC {e_pc:.2e}")

    # ---- 7. RK4 bootstrap matches a full RK4 run for the first steps --------------------
    ts_rk, ys_rk = rk4(f_decay, [1.0], 0, 5, 200)
    check("RK4 solves exponential decay", max_error(ys_rk, ts_rk, exact_decay) < 1e-6)

    # ---- 8. linear invariant: energy of the oscillator stays bounded --------------------
    ts, ys = predictor_corrector(f_ho, [1.0, 0.0], 0, 20, 1000, order=4)
    energies = [0.5 * (y[0] ** 2 + y[1] ** 2) for y in ys]
    check("oscillator energy stays near 0.5", all(abs(e - 0.5) < 0.01 for e in energies),
          f"drift {max(abs(e-0.5) for e in energies):.4f}")

    # ---- 9. edge cases ------------------------------------------------------------------
    ts, ys = predictor_corrector(f_decay, [1.0], 0, 1, 5, order=4)  # few steps, mostly bootstrap
    check("few-step run returns right length", len(ys) == 6)
    try:
        adams_bashforth(f_decay, [1.0], 0, 1, 10, order=7)
        check("bad order raises", False)
    except ValueError:
        check("bad order raises", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
