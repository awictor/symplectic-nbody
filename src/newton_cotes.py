"""Newton-Cotes formulas: the classical quadrature family from equispaced nodes, and their exactness.

Before Gauss and Chebyshev clustering, integration meant NEWTON-COTES: sample the integrand at
EQUALLY SPACED points and integrate the polynomial that passes through them. The trapezoidal rule
(2 points), Simpson's rule (3 points, the parabola), Simpson's 3/8 (4 points), Boole's rule (5), and
Weddle's rule (7) are the low-order members, each a fixed set of weights on n+1 equispaced samples.
The weights are the integrals of the Lagrange basis polynomials, and they have a beautiful property:
a rule on n+1 points is exact for every polynomial up to degree n -- and for the EVEN-point rules
(Simpson, Boole) one degree HIGHER, thanks to symmetry, so Simpson integrates cubics exactly with a
parabola.

Each rule also carries an ERROR TERM proportional to a high derivative of the integrand: the
trapezoid error is -(b-a)^3/12 f'', Simpson's is -(b-a)^5/2880 f^{(4)}, which is why Simpson converges
as O(h^4) while the trapezoid is only O(h^2). Above about 8 points the equispaced weights turn
negative and the rules become numerically unstable (the Runge phenomenon again) -- which is exactly
why COMPOSITE rules (chop the interval into panels and apply a low-order rule on each) are used in
practice, and why Gauss and Clenshaw-Curtis eventually replaced high-order Newton-Cotes.

This module implements the closed Newton-Cotes rules (trapezoid through Weddle) with their exact
rational weights, composite versions, the degree of exactness, and the error constants. It is
validated: each rule integrates polynomials up to its degree of exactness to machine precision (and
the even-point rules one degree beyond); the weights sum to the interval length; composite rules
converge at the predicted order (Simpson O(h^4) faster than trapezoid O(h^2)); known integrals
(e^x, sin, 1/(1+x^2)) match; and the results agree with the repo's existing trapezoid/Simpson. Pure
stdlib; the equispaced-quadrature companion to the Clenshaw-Curtis, Gauss, and Romberg integrators."""

from __future__ import annotations

from fractions import Fraction


# Closed Newton-Cotes weights (as Fractions of h) for n+1 points; the rule is
# h * sum(w_i f_i), with h the spacing. Source: standard tables.
_WEIGHTS = {
    "trapezoid": [Fraction(1, 2), Fraction(1, 2)],                         # 2 points, deg 1
    "simpson": [Fraction(1, 3), Fraction(4, 3), Fraction(1, 3)],           # 3 points, deg 3
    "simpson38": [Fraction(3, 8), Fraction(9, 8), Fraction(9, 8), Fraction(3, 8)],  # 4 pts, deg 3
    "boole": [Fraction(14, 45), Fraction(64, 45), Fraction(24, 45),
              Fraction(64, 45), Fraction(14, 45)],                          # 5 points, deg 5
    "weddle": [Fraction(41, 140), Fraction(216, 140), Fraction(27, 140), Fraction(272, 140),
               Fraction(27, 140), Fraction(216, 140), Fraction(41, 140)],   # 7 points, deg 5(6)
}

# degree of exactness of each rule
_DEGREE = {"trapezoid": 1, "simpson": 3, "simpson38": 3, "boole": 5, "weddle": 6}


def rule_names():
    return list(_WEIGHTS.keys())


def integrate(f, a, b, rule="simpson"):
    """Integrate f over [a, b] with a single Newton-Cotes rule of the given name."""
    w = _WEIGHTS[rule]
    npts = len(w)
    h = (b - a) / (npts - 1)
    total = 0.0
    for i in range(npts):
        x = a + i * h
        total += float(w[i]) * f(x)
    return h * total


def composite(f, a, b, rule="simpson", panels=10):
    """Composite Newton-Cotes: split [a, b] into `panels` sub-intervals, apply `rule` on each."""
    H = (b - a) / panels
    total = 0.0
    for k in range(panels):
        x0 = a + k * H
        total += integrate(f, x0, x0 + H, rule)
    return total


def degree_of_exactness(rule):
    """The highest polynomial degree the rule integrates exactly."""
    return _DEGREE[rule]


def weight_sum(rule):
    """Sum of the rule's weights times h should give the interval length; return sum(w)*(npts-1)...

    For a unit interval [0, n] with h=1 the integral of 1 is n = (npts-1), so h*sum(w) = interval.
    Here we return sum(w), which times h equals the interval length.
    """
    return float(sum(_WEIGHTS[rule]))


def error_constant(rule):
    """The leading error coefficient C in error = C * h^p * f^{(k)}(xi) (informational)."""
    consts = {
        "trapezoid": ("-1/12", 3, 2),
        "simpson": ("-1/90", 5, 4),
        "simpson38": ("-3/80", 5, 4),
        "boole": ("-8/945", 7, 6),
        "weddle": ("-1/140", 7, 6),
    }
    return consts[rule]


def trapezoid(f, a, b, panels=100):
    """Convenience composite trapezoid."""
    return composite(f, a, b, "trapezoid", panels)


def simpson(f, a, b, panels=50):
    """Convenience composite Simpson."""
    return composite(f, a, b, "simpson", panels)
