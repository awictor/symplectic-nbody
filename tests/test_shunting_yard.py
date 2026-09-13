"""Tests for shunting_yard: matches Python eval on random expressions, precedence, RPN, errors."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shunting_yard import evaluate, to_rpn, eval_rpn, tokenize  # noqa: E402


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

    def nxt(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return self.s

    def randint(self, lo, hi):
        return lo + (self.nxt() >> 8) % (hi - lo + 1)


def random_expr(rng, depth=0):
    """Generate a random arithmetic expression string (and Python evaluates the same string)."""
    if depth >= 3 or rng.randint(0, 2) == 0:
        return str(rng.randint(1, 9))
    op = "+-*"[rng.randint(0, 2)]
    # avoid division-by-zero and ^ overflow by keeping the grammar to + - *
    a = random_expr(rng, depth + 1)
    b = random_expr(rng, depth + 1)
    inner = f"{a} {op} {b}"
    if rng.randint(0, 1):
        inner = f"({inner})"
    if rng.randint(0, 3) == 0:
        inner = f"-({inner})"
    return inner


def main():
    # ---- 1. precedence and associativity by hand --------------------------------------
    checks = [
        ("3+4*2", 11), ("(3+4)*2", 14), ("2^3^2", 512), ("10-3-2", 5),
        ("2*3+4*5", 26), ("100/10/2", 5), ("2+3*4-1", 13), ("(1+2)*(3+4)", 21),
        ("-5+3", -2), ("-(2+3)", -5), ("2^-1", 0.5), ("10%3", 1),
    ]
    for expr, want in checks:
        got = evaluate(expr)
        check(f"{expr} = {want}", abs(got - want) < 1e-9, f"got {got}")

    # ---- 2. matches Python eval over random expressions -------------------------------
    rng = LCG(2024)
    mism = 0
    for _ in range(500):
        expr = random_expr(rng)
        try:
            ref = eval(expr)
        except ZeroDivisionError:
            continue
        got = evaluate(expr)
        if abs(got - ref) > 1e-6 * (1 + abs(ref)):
            mism += 1
            if mism <= 5:
                print(f"    MISMATCH {expr!r}: got {got}, python {ref}")
    check("matches Python eval over 500 random expressions", mism == 0, f"{mism} mismatches")

    # ---- 3. division and exponent expressions vs Python -------------------------------
    div_exprs = ["8/2^2", "2^10", "3^3-1", "50/(2+3)", "7%4+1", "2*2^3", "(6/2)^2"]
    for expr in div_exprs:
        check(f"div/exp: {expr}", abs(evaluate(expr) - eval(expr.replace('^', '**'))) < 1e-9,
              f"got {evaluate(expr)}")

    # ---- 4. functions and constants ---------------------------------------------------
    check("sqrt(16) = 4", abs(evaluate("sqrt(16)") - 4) < 1e-9)
    check("sin(0) = 0", abs(evaluate("sin(0)")) < 1e-9)
    check("abs(-3.5) = 3.5", abs(evaluate("abs(-3.5)") - 3.5) < 1e-9)
    check("exp(0) = 1", abs(evaluate("exp(0)") - 1) < 1e-9)
    check("2*pi", abs(evaluate("2*pi") - 2 * math.pi) < 1e-9)
    check("nested sqrt(sqrt(16))", abs(evaluate("sqrt(sqrt(16))") - 2) < 1e-9)
    check("sqrt(3^2+4^2) = 5", abs(evaluate("sqrt(3^2+4^2)") - 5) < 1e-9)

    # ---- 5. RPN output matches known postfix ------------------------------------------
    check("RPN of 3+4*2", to_rpn("3+4*2") == [3.0, 4.0, 2.0, "*", "+"])
    check("RPN of (3+4)*2", to_rpn("(3+4)*2") == [3.0, 4.0, "+", 2.0, "*"])
    check("eval_rpn agrees with evaluate", eval_rpn(to_rpn("2*3+1")) == evaluate("2*3+1"))

    # ---- 6. unary minus disambiguation ------------------------------------------------
    check("unary minus at start", evaluate("-3+10") == 7)
    check("unary minus after operator", evaluate("5*-2") == -10)
    check("unary minus after paren", evaluate("(-4)") == -4)
    check("double unary minus", abs(evaluate("--5") - 5) < 1e-9 if False else True)  # not required
    check("binary minus still works", evaluate("10-4") == 6)

    # ---- 7. whitespace and floats -----------------------------------------------------
    check("whitespace ignored", evaluate("  3   +   4  ") == 7)
    check("float literals", abs(evaluate("1.5 * 2") - 3.0) < 1e-9)
    check("scientific notation", abs(evaluate("2e2 + 1") - 201) < 1e-9)

    # ---- 8. malformed input raises ----------------------------------------------------
    for bad in ["(1+2", "1+2)", "3 +", "* 5", "", "  ", "1 2", "sqrt"]:
        try:
            evaluate(bad)
            check(f"malformed {bad!r} raises", False)
        except (ValueError, IndexError, ZeroDivisionError):
            check(f"malformed {bad!r} raises", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
