"""Tests for DFA minimization: language preserved, minimal state count, equivalence, edge cases."""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dfa_minimization import DFA, minimize, equivalent, brute_minimize_state_count  # noqa: E402


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


def _all_strings(alphabet, max_len):
    for L in range(max_len + 1):
        for combo in itertools.product(alphabet, repeat=L):
            yield "".join(combo)


def _same_language(d1, d2, alphabet, max_len=6):
    return all(d1.accepts(s) == d2.accepts(s) for s in _all_strings(alphabet, max_len))


def main():
    # ---- 1. classic redundant DFA: "ends in a" over {a,b}, with extra states ------------
    # states 0..3; 0 start; 1 and 3 both "just saw a" (redundant); accept on seeing a
    # Build a DFA with duplicate structure to be merged.
    d = DFA(
        states={0, 1, 2, 3},
        alphabet=["a", "b"],
        transitions={
            (0, "a"): 1, (0, "b"): 2,
            (1, "a"): 3, (1, "b"): 2,
            (2, "a"): 1, (2, "b"): 2,
            (3, "a"): 3, (3, "b"): 2,
        },
        start=0,
        accepting={1, 3},
    )
    m = minimize(d)
    check("ends-in-a: language preserved", _same_language(d, m, "ab"))
    check("ends-in-a: minimized to 2 states", len(m.states) == 2, f"{len(m.states)}")
    check("ends-in-a: matches brute count", len(m.states) == brute_minimize_state_count(d))

    # ---- 2. random DFAs: minimization preserves language + matches brute count ----------
    def _lcg(seed):
        st = seed & 0xFFFFFFFF

        def nxt():
            nonlocal st
            st = (1664525 * st + 1013904223) & 0xFFFFFFFF
            return st >> 8
        return nxt

    rng = _lcg(2024)
    lang_ok = count_ok = 0
    trials = 200
    for _ in range(trials):
        n = 2 + rng() % 6
        alpha = ["0", "1"]
        trans = {}
        for s in range(n):
            for c in alpha:
                trans[(s, c)] = rng() % n
        acc = {s for s in range(n) if rng() % 2}
        d = DFA(set(range(n)), alpha, trans, 0, acc)
        m = minimize(d)
        if _same_language(d, m, "01", max_len=7):
            lang_ok += 1
        if len(m.states) == brute_minimize_state_count(d):
            count_ok += 1
    check("random DFAs: language preserved (200)", lang_ok == trials, f"{lang_ok}/{trials}")
    check("random DFAs: minimal state count == brute (200)", count_ok == trials,
          f"{count_ok}/{trials}")

    # ---- 3. minimizing an already-minimal DFA is idempotent -----------------------------
    d = DFA({0, 1}, ["a", "b"], {(0, "a"): 1, (0, "b"): 0, (1, "a"): 1, (1, "b"): 0}, 0, {1})
    m1 = minimize(d)
    m2 = minimize(m1)
    check("idempotent: minimizing twice = once", len(m1.states) == len(m2.states) == len(d.states))

    # ---- 4. equivalence of differently-built DFAs ---------------------------------------
    # two DFAs for "contains an even number of a's" over {a,b}
    even_a1 = DFA({0, 1}, ["a", "b"],
                  {(0, "a"): 1, (0, "b"): 0, (1, "a"): 0, (1, "b"): 1}, 0, {0})
    # a redundant 4-state version
    even_a2 = DFA({0, 1, 2, 3}, ["a", "b"],
                  {(0, "a"): 1, (0, "b"): 2, (1, "a"): 2, (1, "b"): 3,
                   (2, "a"): 3, (2, "b"): 0, (3, "a"): 0, (3, "b"): 1}, 0, {0, 2})
    check("equivalent DFAs detected (even #a)", equivalent(even_a1, even_a2))
    check("both minimize to 2 states", len(minimize(even_a2).states) == 2)

    # ---- 5. non-equivalent DFAs ---------------------------------------------------------
    odd_a = DFA({0, 1}, ["a", "b"],
                {(0, "a"): 1, (0, "b"): 0, (1, "a"): 0, (1, "b"): 1}, 0, {1})  # odd #a
    check("even vs odd #a not equivalent", not equivalent(even_a1, odd_a))

    # ---- 6. unreachable states removed --------------------------------------------------
    d = DFA({0, 1, 2}, ["a"], {(0, "a"): 0, (1, "a"): 2, (2, "a"): 1}, 0, {0})
    m = minimize(d)  # states 1,2 unreachable
    check("unreachable states pruned", len(m.states) == 1, f"{len(m.states)}")

    # ---- 7. accept-all and accept-none --------------------------------------------------
    accept_all = DFA({0, 1}, ["a"], {(0, "a"): 1, (1, "a"): 0}, 0, {0, 1})
    m = minimize(accept_all)
    check("accept-all minimizes to 1 state", len(m.states) == 1)
    check("accept-all still accepts everything", all(m.accepts(s) for s in _all_strings("a", 5)))

    accept_none = DFA({0, 1}, ["a"], {(0, "a"): 1, (1, "a"): 0}, 0, set())
    m = minimize(accept_none)
    check("accept-none accepts nothing", not any(m.accepts(s) for s in _all_strings("a", 5)))

    # ---- 8. language with a dead state: "exactly one a" ---------------------------------
    # 0 -a-> 1 (accept), 1 -a-> dead, b loops; strings with exactly one 'a'
    d = DFA({0, 1, 2}, ["a", "b"],
            {(0, "a"): 1, (0, "b"): 0, (1, "a"): 2, (1, "b"): 1, (2, "a"): 2, (2, "b"): 2},
            0, {1})
    m = minimize(d)
    check("exactly-one-a: language preserved", _same_language(d, m, "ab"))
    check("exactly-one-a: accepts 'a','ba','ab'",
          m.accepts("a") and m.accepts("ba") and m.accepts("bab"))
    check("exactly-one-a: rejects '', 'aa'", not m.accepts("") and not m.accepts("aa"))

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
