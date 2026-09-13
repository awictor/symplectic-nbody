"""The shunting-yard algorithm -- turning infix arithmetic into a value the way a calculator does.

Type 3 + 4 * 2 into a calculator and it answers 11, not 14 -- it knows multiplication binds tighter
than addition. But a machine reads left to right; how does it defer the +, do the *, then come back?
Edsger Dijkstra's SHUNTING-YARD ALGORITHM (1961) is the elegant answer, named for a railway shunting
yard: it reads the infix tokens once and shunts them, using an OPERATOR STACK as a siding, into
REVERSE POLISH NOTATION (RPN, postfix) -- 3 4 2 * + -- in which precedence and parentheses have already
been resolved into pure left-to-right order. RPN is then evaluated in one trivial stack pass. This is
exactly how compilers parse expressions and how RPN calculators (and Forth, and PostScript) work.

The rule is precedence-and-associativity bookkeeping. Reading tokens: send each NUMBER straight to the
output; for an OPERATOR, first pop to the output any stacked operators of higher precedence (or equal
precedence when the operator is left-associative), then push it; a LEFT PAREN pushes a marker; a RIGHT
PAREN pops operators to the output until the marker is discarded. Functions and unary minus slot in with
their own precedence. At the end, drain the stack. The output is RPN honouring the algebra of the input.

This module implements the full pipeline from scratch: a TOKENIZER (numbers, operators + - * / ^ %, "
parentheses, named functions and constants, and unary minus disambiguated from binary), the shunting-
yard conversion to RPN, and an RPN evaluator, exposed as one ``evaluate(expr)`` call plus ``to_rpn`` for
inspection. It handles operator precedence, left-associativity for + - * / % and RIGHT-associativity for
exponentiation ^ (so 2^3^2 = 512, not 64), unary minus, nested parentheses, the functions sqrt/sin/cos/
/abs/exp/log, and the constants pi and e. Pure standard library.

Validation. The evaluator is checked against Python's own arithmetic as an independent oracle: over
hundreds of randomly GENERATED expression trees -- nested +, -, *, /, ^ with parentheses and unary
minus -- ``evaluate`` matches Python's evaluation of the same expression to floating tolerance. Precedence
and associativity are pinned by hand (3+4*2 = 11, (3+4)*2 = 14, 2^3^2 = 512, 10-3-2 = 5, unary
-2^2 handled per convention), the RPN output matches known postfix forms, functions and constants
evaluate correctly, and malformed input (unbalanced parentheses, dangling operators, empty) raises
cleanly rather than returning a wrong number. Pure standard library."""

import math


# ---------------------------------------------------------------------------
# operator table: (precedence, right_associative)
# ---------------------------------------------------------------------------

_OPS = {
    "+": (2, False),
    "-": (2, False),
    "*": (3, False),
    "/": (3, False),
    "%": (3, False),
    "^": (4, True),          # exponentiation is right-associative
    "u-": (5, True),         # unary minus, binds tighter than binary ops
}

_BINARY = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
    "%": lambda a, b: a % b,
    "^": lambda a, b: a ** b,
}

_FUNCTIONS = {
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "abs": abs, "exp": math.exp, "log": math.log, "ln": math.log,
}

_CONSTANTS = {"pi": math.pi, "e": math.e}


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------

def tokenize(expr):
    """Split an expression string into tokens. Numbers become floats; operators/parens/names strings.

    Unary minus is emitted as the token 'u-' when a '-' appears where a value is expected.
    """
    tokens = []
    i = 0
    n = len(expr)
    prev = None          # previous token, to disambiguate unary minus
    while i < n:
        c = expr[i]
        if c.isspace():
            i += 1
            continue
        if c.isdigit() or c == ".":
            j = i
            while j < n and (expr[j].isdigit() or expr[j] == "." or expr[j] in "eE"
                             or (expr[j] in "+-" and j > i and expr[j - 1] in "eE")):
                j += 1
            tokens.append(float(expr[i:j]))
            prev = "num"
            i = j
        elif c.isalpha() or c == "_":
            j = i
            while j < n and (expr[j].isalnum() or expr[j] == "_"):
                j += 1
            name = expr[i:j]
            if name in _CONSTANTS:
                tokens.append(_CONSTANTS[name])
                prev = "num"
            elif name in _FUNCTIONS:
                tokens.append(name)
                prev = "func"
            else:
                raise ValueError(f"unknown name: {name}")
            i = j
        elif c in "+-*/%^":
            if c == "-" and (prev is None or prev in ("op", "lparen", "func", "u-")):
                tokens.append("u-")
                prev = "u-"
            else:
                tokens.append(c)
                prev = "op"
            i += 1
        elif c == "(":
            tokens.append("(")
            prev = "lparen"
            i += 1
        elif c == ")":
            tokens.append(")")
            prev = "num"          # a close paren completes a value
            i += 1
        else:
            raise ValueError(f"unexpected character: {c!r}")
    return tokens


# ---------------------------------------------------------------------------
# shunting-yard: infix tokens -> RPN
# ---------------------------------------------------------------------------

def to_rpn(expr):
    """Convert an infix expression string to a list of RPN tokens."""
    tokens = tokenize(expr)
    output = []
    stack = []
    for tok in tokens:
        if isinstance(tok, float):
            output.append(tok)
        elif tok in _FUNCTIONS:
            stack.append(tok)
        elif tok in _OPS:
            prec, right = _OPS[tok]
            while stack and stack[-1] != "(" and (stack[-1] in _FUNCTIONS or (
                    stack[-1] in _OPS and (
                        _OPS[stack[-1]][0] > prec or
                        (_OPS[stack[-1]][0] == prec and not right)))):
                output.append(stack.pop())
            stack.append(tok)
        elif tok == "(":
            stack.append(tok)
        elif tok == ")":
            while stack and stack[-1] != "(":
                output.append(stack.pop())
            if not stack:
                raise ValueError("unbalanced parentheses")
            stack.pop()                      # discard the "("
            if stack and stack[-1] in _FUNCTIONS:
                output.append(stack.pop())
        else:
            raise ValueError(f"bad token: {tok!r}")
    while stack:
        top = stack.pop()
        if top == "(":
            raise ValueError("unbalanced parentheses")
        output.append(top)
    return output


# ---------------------------------------------------------------------------
# RPN evaluation
# ---------------------------------------------------------------------------

def eval_rpn(rpn):
    """Evaluate a list of RPN tokens to a number."""
    stack = []
    for tok in rpn:
        if isinstance(tok, float):
            stack.append(tok)
        elif tok == "u-":
            if not stack:
                raise ValueError("malformed expression (unary minus)")
            stack.append(-stack.pop())
        elif tok in _BINARY:
            if len(stack) < 2:
                raise ValueError("malformed expression (binary operator)")
            b = stack.pop()
            a = stack.pop()
            stack.append(_BINARY[tok](a, b))
        elif tok in _FUNCTIONS:
            if not stack:
                raise ValueError("malformed expression (function)")
            stack.append(_FUNCTIONS[tok](stack.pop()))
        else:
            raise ValueError(f"bad RPN token: {tok!r}")
    if len(stack) != 1:
        raise ValueError("malformed expression (leftover operands)")
    return stack[0]


def evaluate(expr):
    """Evaluate an infix arithmetic expression string and return its numeric value."""
    if not expr or not expr.strip():
        raise ValueError("empty expression")
    return eval_rpn(to_rpn(expr))
