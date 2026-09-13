"""Tests for string similarity: Jaro/Winkler textbook values, Jaccard/Dice properties, Soundex codes."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from string_similarity import (  # noqa: E402
    jaro,
    jaro_winkler,
    jaccard,
    dice,
    soundex,
    qgrams,
    rank,
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
    # ---- 1. Jaro textbook values --------------------------------------------------------
    check("Jaro(MARTHA, MARHTA) = 0.944", abs(jaro("MARTHA", "MARHTA") - 0.9444444) < 1e-4,
          f"{jaro('MARTHA', 'MARHTA'):.4f}")
    check("Jaro(DWAYNE, DUANE) = 0.822", abs(jaro("DWAYNE", "DUANE") - 0.8222222) < 1e-4,
          f"{jaro('DWAYNE', 'DUANE'):.4f}")
    check("Jaro(DIXON, DICKSONX) = 0.767", abs(jaro("DIXON", "DICKSONX") - 0.7666666) < 1e-4,
          f"{jaro('DIXON', 'DICKSONX'):.4f}")

    # ---- 2. Jaro-Winkler textbook values ------------------------------------------------
    check("JW(MARTHA, MARHTA) = 0.961", abs(jaro_winkler("MARTHA", "MARHTA") - 0.9611111) < 1e-4,
          f"{jaro_winkler('MARTHA', 'MARHTA'):.4f}")
    check("JW(DWAYNE, DUANE) = 0.84", abs(jaro_winkler("DWAYNE", "DUANE") - 0.84) < 1e-4,
          f"{jaro_winkler('DWAYNE', 'DUANE'):.4f}")

    # ---- 3. identity and disjoint -------------------------------------------------------
    check("Jaro identical = 1", jaro("hello", "hello") == 1.0)
    check("JW identical = 1", jaro_winkler("hello", "hello") == 1.0)
    check("Jaro disjoint = 0", jaro("abc", "xyz") == 0.0)
    check("empty vs nonempty = 0", jaro("", "abc") == 0.0)

    # ---- 4. Winkler >= Jaro, boosts only on shared prefix -------------------------------
    ok = True
    pairs = [("martha", "marhta"), ("dwayne", "duane"), ("abcde", "abxyz"), ("zzz", "azz")]
    for a, b in pairs:
        if jaro_winkler(a, b) < jaro(a, b) - 1e-12:
            ok = False
    check("Jaro-Winkler >= Jaro always", ok)
    # no shared prefix -> no boost
    check("no prefix -> JW == Jaro", abs(jaro_winkler("xyz", "abc") - jaro("xyz", "abc")) < 1e-12)

    # ---- 5. Jaccard / Dice properties ---------------------------------------------------
    check("Jaccard symmetric", abs(jaccard("night", "nacht") - jaccard("nacht", "night")) < 1e-12)
    check("Jaccard identical = 1", jaccard("hello", "hello") == 1.0)
    check("Dice identical = 1", dice("hello", "hello") == 1.0)
    check("Jaccard disjoint = 0", jaccard("abc", "xyz") == 0.0)
    # Dice >= Jaccard always (2x/(x+y) >= x/y form)
    ok = True
    for a, b in [("night", "nacht"), ("hello", "help"), ("kitten", "sitting")]:
        if dice(a, b) < jaccard(a, b) - 1e-12:
            ok = False
    check("Dice >= Jaccard", ok)

    # ---- 6. Soundex reference codes -----------------------------------------------------
    check("Soundex(Robert) = R163", soundex("Robert") == "R163", soundex("Robert"))
    check("Soundex(Rupert) = R163", soundex("Rupert") == "R163", soundex("Rupert"))
    check("Soundex(Rubin) = R150", soundex("Rubin") == "R150", soundex("Rubin"))
    check("Soundex(Tymczak) = T522", soundex("Tymczak") == "T522", soundex("Tymczak"))
    check("Soundex(Ashcraft) = A261", soundex("Ashcraft") == "A261", soundex("Ashcraft"))
    check("Soundex(Pfister) = P236", soundex("Pfister") == "P236", soundex("Pfister"))
    check("Soundex(Honeyman) = H555", soundex("Honeyman") == "H555", soundex("Honeyman"))

    # ---- 7. Soundex homophones collide --------------------------------------------------
    check("Robert and Rupert collide", soundex("Robert") == soundex("Rupert"))
    check("different-sounding names differ", soundex("Robert") != soundex("Smith"))

    # ---- 8. ranking finds the closest name ----------------------------------------------
    names = ["Jonathan", "Johnathan", "Jon", "Nathan", "Jonas"]
    ranked = rank("Johnathon", names, method="jaro_winkler")
    check("closest match to 'Johnathon' is Johnathan or Jonathan",
          ranked[0][0] in ("Johnathan", "Jonathan"), ranked[0][0])
    check("ranking is sorted descending", all(ranked[i][1] >= ranked[i + 1][1]
                                              for i in range(len(ranked) - 1)))

    # ---- 9. q-gram helper ---------------------------------------------------------------
    g = qgrams("ab", q=2)
    check("qgrams pads short strings", "#a" in g and "b$" in g)

    # ---- 10. edge cases -----------------------------------------------------------------
    check("Soundex empty", soundex("") == "")
    check("Soundex non-alpha stripped", soundex("O'Brien") == soundex("OBrien"))
    check("Jaccard both empty = 1", jaccard("", "") == 1.0)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
