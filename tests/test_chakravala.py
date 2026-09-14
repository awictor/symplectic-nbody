"""Tests for the chakravala method: agreement with the CF Pell solver, exact verification, composition."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chakravala as CH  # noqa: E402
import pell  # noqa: E402


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
    # ---- 1. known fundamental solutions -------------------------------------------------
    known = {2: (3, 2), 3: (2, 1), 5: (9, 4), 7: (8, 3), 13: (649, 180)}
    for D, (x, y) in known.items():
        check(f"D={D}: chakravala fundamental solution", CH.fundamental_solution(D) == (x, y),
              f"{CH.fundamental_solution(D)}")

    # ---- 2. the classic hard cases D=61 and D=109 --------------------------------------
    x61, y61 = CH.fundamental_solution(61)
    check("D=61: x = 1766319049", x61 == 1766319049 and y61 == 226153980, f"{(x61, y61)}")
    x109, y109 = CH.fundamental_solution(109)
    check("D=109: matches known solution",
          x109 == 158070671986249 and y109 == 15140424455100, f"{(x109, y109)}")

    # ---- 3. the fundamental solution satisfies x^2 - D y^2 = 1 exactly ------------------
    ok = True
    for D in range(2, 60):
        if CH.is_square(D):
            continue
        x, y = CH.fundamental_solution(D)
        if not CH.verify(D, x, y):
            ok = False
            break
    check("fundamental solution satisfies the equation exactly (D<60)", ok)

    # ---- 4. chakravala agrees with the continued-fraction Pell solver for all D<=150 ----
    mismatches = 0
    for D in range(2, 151):
        if CH.is_square(D):
            continue
        if CH.fundamental_solution(D) != pell.fundamental_solution(D):
            mismatches += 1
    check("chakravala == CF Pell for all non-square D<=150", mismatches == 0, f"{mismatches}")

    # ---- 5. perfect squares raise (no solution) -----------------------------------------
    raised = 0
    for sq in (4, 9, 16, 25, 100):
        try:
            CH.fundamental_solution(sq)
        except ValueError:
            raised += 1
    check("perfect squares raise ValueError", raised == 5)

    # ---- 6. Brahmagupta composition produces further solutions --------------------------
    sols = CH.solutions(7, 5)
    check("D=7: 5 generated solutions all satisfy the equation",
          all(CH.verify(7, x, y) for x, y in sols), f"{sols}")
    check("D=7: solutions are strictly increasing", all(sols[i][0] < sols[i + 1][0]
                                                         for i in range(len(sols) - 1)))

    # ---- 7. composition matches the recurrence for a larger D ---------------------------
    x1, y1 = CH.fundamental_solution(13)
    second = CH.compose((x1, y1), (x1, y1), 13)
    check("D=13: composed second solution is valid", CH.verify(13, *second), f"{second}")

    # ---- 8. every intermediate triple satisfies its a^2 - D b^2 = k invariant -----------
    D = 61
    x, y, triples = CH.fundamental_solution(D, trace=True)
    check("all intermediate triples satisfy a^2 - D b^2 = k",
          all(a * a - D * b * b == k for a, b, k in triples), f"{triples}")
    check("trace ends at k=1", triples[-1][2] == 1)

    # ---- 9. arbitrary-precision: a very large solution stays exact -----------------------
    # D = 94 has a large fundamental solution
    x94, y94 = CH.fundamental_solution(94)
    check("D=94 solution exact (large integers)", CH.verify(94, x94, y94),
          f"x has {len(str(x94))} digits")
    check("D=94 matches Pell solver", (x94, y94) == pell.fundamental_solution(94))

    # ---- 10. small non-square D exhaustive check ----------------------------------------
    allok = True
    for D in range(2, 40):
        if CH.is_square(D):
            continue
        x, y = CH.fundamental_solution(D)
        if x * x - D * y * y != 1:
            allok = False
    check("exhaustive: D=2..39 all give valid fundamental solutions", allok)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
