"""Tests for Wang-Landau: recovered ln g matches exact enumeration, thermodynamics agree, sums to 2^N."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wang_landau import (  # noqa: E402
    wang_landau,
    exact_density_of_states,
    thermodynamics,
    partition_function_lse,
    _allowed_energies,
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
    # ---- exact density of states for the 3x3 and 4x4 Ising lattice ----------------------
    n = 3
    ex_e, ex_lng, ex_counts = exact_density_of_states(n)

    # ---- 1. exact g sums to 2^N ---------------------------------------------------------
    check("exact g sums to 2^N", sum(ex_counts) == (1 << (n * n)), f"{sum(ex_counts)}")

    # ---- 2. exact ground state is doubly degenerate (all-up, all-down) ------------------
    check("exact ground state degeneracy 2", ex_counts[0] == 2, f"{ex_counts[0]}")

    # ---- 3. Wang-Landau recovers ln g matching exact (up to overall constant) -----------
    wl_e, wl_lng = wang_landau(n, seed=1, flat=0.8, f_final=1e-4, max_sweeps=60000)
    # compare on populated bins; both normalized so ground state = ln 2
    max_rel = 0.0
    for i in range(len(ex_e)):
        if ex_lng[i] != float("-inf") and wl_lng[i] != float("-inf"):
            diff = abs(wl_lng[i] - ex_lng[i])
            denom = abs(ex_lng[i]) if abs(ex_lng[i]) > 1e-9 else 1.0
            max_rel = max(max_rel, diff / denom)
    check("WL ln g matches exact within 8%", max_rel < 0.08, f"max rel {max_rel:.3%}")

    # ---- 4. WL populates the same energy bins the exact DOS does ------------------------
    ex_pop = {i for i in range(len(ex_e)) if ex_lng[i] != float("-inf")}
    wl_pop = {i for i in range(len(wl_e)) if wl_lng[i] != float("-inf")}
    check("WL populates all attainable energies", ex_pop.issubset(wl_pop),
          f"missing {ex_pop - wl_pop}")

    # ---- 5. total states recovered ~ 2^N ------------------------------------------------
    # sum g(E) = sum exp(ln g). Compare log of total to N ln 2.
    lse = partition_function_lse(wl_e, wl_lng, temperature=1e12)  # beta->0: Z = sum g
    check("WL total states ~ 2^N", abs(lse - (n * n) * math.log(2)) < 0.08 * (n * n) * math.log(2),
          f"lnZ {lse:.3f} vs {(n*n)*math.log(2):.3f}")

    # ---- 6. thermodynamics from WL agree with thermodynamics from exact g ---------------
    ok = True
    for T in [1.0, 2.0, 2.27, 3.0, 5.0]:
        e_exact, c_exact = thermodynamics(ex_e, ex_lng, T)
        e_wl, c_wl = thermodynamics(wl_e, wl_lng, T)
        # internal energy per bond scale ~ O(N); require close absolute agreement
        if abs(e_wl - e_exact) > 0.5 or abs(c_wl - c_exact) > max(1.0, 0.15 * abs(c_exact)):
            ok = False
            check("WL thermodynamics ~ exact", False, f"T={T}: E {e_wl:.3f}/{e_exact:.3f} C {c_wl:.3f}/{c_exact:.3f}")
            break
    if ok:
        check("WL <E> and C match exact across temperatures", True)

    # ---- 7. internal energy is monotone increasing with temperature ---------------------
    Es = [thermodynamics(ex_e, ex_lng, T)[0] for T in [0.5, 1.0, 2.0, 4.0, 10.0]]
    check("exact <E> increases with T", all(Es[i + 1] >= Es[i] - 1e-9 for i in range(len(Es) - 1)),
          f"{[round(e,2) for e in Es]}")

    # ---- 8. specific heat is non-negative -----------------------------------------------
    Cs = [thermodynamics(ex_e, ex_lng, T)[1] for T in [0.5, 1.0, 2.0, 2.27, 3.0, 5.0]]
    check("specific heat non-negative", all(c >= -1e-9 for c in Cs), f"{[round(c,3) for c in Cs]}")

    # ---- 9. allowed-energy grid steps by 4 and brackets +/- 2*bonds ---------------------
    grid = _allowed_energies(3)
    bonds = 2 * 9
    check("energy grid endpoints", grid[0] == -bonds and grid[-1] == bonds)
    check("energy grid step 4", all(grid[i + 1] - grid[i] == 4 for i in range(len(grid) - 1)))

    # ---- 10. 4x4 lattice: WL still recovers the DOS shape (heavier, so looser) ----------
    ex4_e, ex4_lng, ex4_counts = exact_density_of_states(4)
    check("4x4 exact sums to 2^16", sum(ex4_counts) == (1 << 16))
    wl4_e, wl4_lng = wang_landau(4, seed=3, flat=0.8, f_final=1e-5, max_sweeps=250000)
    # use ABSOLUTE ln-g tolerance: the extreme high-energy bins have tiny ln g (~0.7) so a small
    # absolute error there is a large relative error, yet contributes negligibly to any observable.
    max_abs4 = 0.0
    for i in range(len(ex4_e)):
        if ex4_lng[i] != float("-inf") and wl4_lng[i] != float("-inf"):
            max_abs4 = max(max_abs4, abs(wl4_lng[i] - ex4_lng[i]))
    check("4x4 WL ln g within 0.2 absolute", max_abs4 < 0.2, f"max abs {max_abs4:.3f}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
