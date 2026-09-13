"""Tests for CYK parsing: membership + parse count match brute derivation, classic grammars."""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cyk import (  # noqa: E402
    CNFGrammar,
    recognize,
    count_parses,
    parse_tree,
    tree_yield,
    brute_derivations,
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
    # ---- 1. balanced parentheses grammar ------------------------------------------------
    # S -> S S | ( S ) | ()   in CNF:
    #   S -> S S
    #   S -> L R      (L='(', R=')')  covers "()"
    #   S -> L T      T -> S R        covers "( S )"
    #   L -> (        R -> )
    paren = CNFGrammar(
        binary_rules=[
            ("S", "S", "S"),
            ("S", "L", "R"),
            ("S", "L", "T"),
            ("T", "S", "R"),
        ],
        terminal_rules=[("L", "("), ("R", ")")],
        start="S",
    )
    check("() balanced", recognize(paren, "()"))
    check("(()) balanced", recognize(paren, "(())"))
    check("()() balanced", recognize(paren, "()()"))
    check("(()()) balanced", recognize(paren, "(()())"))
    check("( unbalanced", not recognize(paren, "("))
    check(") unbalanced", not recognize(paren, ")"))
    check("(() unbalanced", not recognize(paren, "(()"))
    check(")( unbalanced", not recognize(paren, ")("))

    # ---- 2. membership matches brute derivation over all short strings ------------------
    mism = 0
    for L in range(1, 9):
        for s in itertools.product("()", repeat=L):
            s = "".join(s)
            if recognize(paren, s) != (brute_derivations(paren, s) > 0):
                mism += 1
    check("membership == brute derivation (all strings len<=8)", mism == 0, f"{mism}")

    # ---- 3. parse count matches brute derivation count ----------------------------------
    mism = 0
    for L in range(1, 9):
        for s in itertools.product("()", repeat=L):
            s = "".join(s)
            if count_parses(paren, s) != brute_derivations(paren, s):
                mism += 1
    check("count_parses == brute derivation count (len<=8)", mism == 0, f"{mism}")

    # ---- 4. a^n b^n grammar -------------------------------------------------------------
    # S -> A B | A C ; C -> S B ; A -> a ; B -> b   (S derives a^n b^n, n>=1)
    anbn = CNFGrammar(
        binary_rules=[("S", "A", "B"), ("S", "A", "C"), ("C", "S", "B")],
        terminal_rules=[("A", "a"), ("B", "b")],
        start="S",
    )
    check("ab in a^n b^n", recognize(anbn, "ab"))
    check("aabb in a^n b^n", recognize(anbn, "aabb"))
    check("aaabbb in a^n b^n", recognize(anbn, "aaabbb"))
    check("aab NOT in a^n b^n", not recognize(anbn, "aab"))
    check("abb NOT in a^n b^n", not recognize(anbn, "abb"))
    check("ba NOT in a^n b^n", not recognize(anbn, "ba"))
    check("a^n b^n unambiguous (one parse)", all(count_parses(anbn, "a" * k + "b" * k) == 1
                                                 for k in range(1, 6)))

    # ---- 5. even-length palindromes over {a,b} ------------------------------------------
    # S -> A A' | B B' ... simpler: S -> a S a | b S b | aa | bb (even palindromes)
    # CNF: S -> A X (X -> S A) | A A ; and same for b
    palin = CNFGrammar(
        binary_rules=[
            ("S", "A", "Xa"), ("Xa", "S", "A"), ("S", "A", "A"),
            ("S", "B", "Xb"), ("Xb", "S", "B"), ("S", "B", "B"),
        ],
        terminal_rules=[("A", "a"), ("B", "b")],
        start="S",
    )
    check("aa is even palindrome", recognize(palin, "aa"))
    check("abba is even palindrome", recognize(palin, "abba"))
    check("abccba... aabbaa is even palindrome", recognize(palin, "aabbaa"))
    check("ab NOT palindrome", not recognize(palin, "ab"))
    check("aab NOT even-length", not recognize(palin, "aab"))
    check("abab NOT palindrome", not recognize(palin, "abab"))

    # ---- 6. parse tree yield reproduces the input ---------------------------------------
    for s in ["()", "(())", "()()", "(()())"]:
        t = parse_tree(paren, s)
        check(f"parse tree of '{s}' yields the input", t is not None and "".join(tree_yield(t)) == s)
    check("no parse tree for invalid string", parse_tree(paren, "(()") is None)

    # ---- 7. ambiguity: a grammar with multiple parses -----------------------------------
    # S -> S S | a  is ambiguous for aaa (two ways to associate)
    amb = CNFGrammar(binary_rules=[("S", "S", "S")], terminal_rules=[("S", "a")], start="S")
    check("aaa has multiple parses (ambiguous)", count_parses(amb, "aaa") == 2)
    check("aaaa has 5 parses (Catalan C_3)", count_parses(amb, "aaaa") == 5)
    check("single a has one parse", count_parses(amb, "a") == 1)
    # Catalan numbers: number of parses of a^n is C_{n-1}
    catalan = [1, 1, 2, 5, 14, 42]
    check("a^n parse counts are Catalan numbers",
          all(count_parses(amb, "a" * n) == catalan[n - 1] for n in range(1, 6)))

    # ---- 8. edge cases ------------------------------------------------------------------
    check("empty string rejected (no epsilon)", not recognize(paren, ""))
    check("empty string zero parses", count_parses(paren, "") == 0)
    check("unknown terminal rejected", not recognize(paren, "x"))

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
