"""Demo: the shunting-yard algorithm -- infix expressions to RPN to a value.

Evaluates several expressions, showing the RPN each produces and confirming precedence and
associativity, and checks against Python's own arithmetic. Draws the infix -> RPN shunting of a
sample expression.

    python examples/shunting_yard_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shunting_yard import evaluate, to_rpn, tokenize  # noqa: E402


def fmt_rpn(rpn):
    return " ".join(str(int(t)) if isinstance(t, float) and t == int(t) else str(t) for t in rpn)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Shunting-yard: read infix once, shunt to RPN, evaluate in one pass\n")

    exprs = [
        "3 + 4 * 2",
        "(3 + 4) * 2",
        "2 ^ 3 ^ 2",
        "10 - 3 - 2",
        "sqrt(3^2 + 4^2)",
        "-5 * (2 + 3)",
        "100 / 10 / 2",
    ]
    print(f"  {'infix':<20} {'RPN (postfix)':<24} {'value':>10}   Python")
    for e in exprs:
        rpn = fmt_rpn(to_rpn(e))
        val = evaluate(e)
        try:
            py = eval(e.replace("^", "**").replace("sqrt", "__import__('math').sqrt"))
            match = abs(val - py) < 1e-9
        except Exception:
            match = "-"
        print(f"  {e:<20} {rpn:<24} {val:>10.4g}   {match}")
    print()
    print("  Note 2^3^2 = 512, not 64: exponentiation is RIGHT-associative, so it groups as 2^(3^2).")
    print("  And 3+4*2 = 11, not 14: the * is popped before the + because it binds tighter.\n")

    print("  Each operator waits on a stack until an operator of equal-or-lower precedence arrives,")
    print("  then it is shunted to the output -- so the RPN comes out with precedence and parentheses")
    print("  already baked into left-to-right order, ready for a trivial stack evaluation. This is how")
    print("  calculators, compilers, and RPN languages like Forth and PostScript parse arithmetic.")

    _svg(os.path.join(outdir, "shunting_yard.svg"), "3 + 4 * 2 + 1")
    print(f"\n  wrote {os.path.join(outdir, 'shunting_yard.svg')}")


def _svg(path, expr, width=760, height=380):
    # replay shunting-yard, capturing the (output, stack) at each step
    tokens = tokenize(expr)
    from shunting_yard import _OPS, _FUNCTIONS
    output, stack = [], []
    steps = []
    for tok in tokens:
        if isinstance(tok, float):
            output.append(tok)
        elif tok in _OPS:
            prec, right = _OPS[tok]
            while stack and stack[-1] != "(" and stack[-1] in _OPS and (
                    _OPS[stack[-1]][0] > prec or (_OPS[stack[-1]][0] == prec and not right)):
                output.append(stack.pop())
            stack.append(tok)
        elif tok == "(":
            stack.append(tok)
        elif tok == ")":
            while stack and stack[-1] != "(":
                output.append(stack.pop())
            if stack:
                stack.pop()
        steps.append((str(_tok(tok)), [_tok(o) for o in output], [_tok(s) for s in stack]))
    while stack:
        output.append(stack.pop())
    steps.append(("(drain)", [_tok(o) for o in output], []))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="16">'
        f'Shunting {expr!r} to RPN: output queue and operator stack per token</text>',
        f'<text x="30" y="58" fill="#8b949e" font-size="12">token</text>',
        f'<text x="150" y="58" fill="#06d6a0" font-size="12">output (RPN so far)</text>',
        f'<text x="480" y="58" fill="#ffd43b" font-size="12">operator stack</text>',
    ]
    y = 80
    for tok, out, stk in steps:
        parts.append(f'<text x="30" y="{y}" fill="#4dabf7" font-size="13">{tok}</text>')
        parts.append(f'<text x="150" y="{y}" fill="#06d6a0" font-size="13">{" ".join(out)}</text>')
        parts.append(f'<text x="480" y="{y}" fill="#ffd43b" font-size="13">{" ".join(stk)}</text>')
        y += 26
    parts.append(f'<text x="30" y="{y+10}" fill="#8b949e" font-size="12">'
                 f'final RPN = {" ".join(_tok(o) for o in output)}  ->  evaluates to '
                 f'{evaluate(expr):.0f}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


def _tok(t):
    if isinstance(t, float):
        return str(int(t)) if t == int(t) else str(t)
    return str(t)


if __name__ == "__main__":
    main()
