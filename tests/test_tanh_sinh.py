"""Tests for tanh_sinh: double-exponential quadrature vs closed-form integrals, incl. singular ones."""

import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tanh_sinh import integrate

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def close(a, b, tol=1e-7):
    return abs(a - b) <= tol * max(1.0, abs(b))


def _simpson_ref(f, a, b, n):
    if n % 2:
        n += 1
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * f(a + i * h)
    return s * h / 3


# --- smooth integrands (should be near machine precision) ------------------
check("integral of x^2 on [0,1] = 1/3", close(integrate(lambda x: x * x, 0, 1), 1 / 3, 1e-12))
check("integral of sin on [0,pi] = 2", close(integrate(math.sin, 0, math.pi), 2, 1e-12))
check("integral of exp on [0,1] = e-1", close(integrate(math.exp, 0, 1), math.e - 1, 1e-12))
check("integral of 1/(1+x^2) on [0,1] = pi/4",
      close(integrate(lambda x: 1 / (1 + x * x), 0, 1), math.pi / 4, 1e-12))
check("integral of a cubic on [-2,3]",
      close(integrate(lambda x: x ** 3 - 2 * x + 1, -2, 3),
            (3 ** 4 / 4 - 3 ** 2 + 3) - ((-2) ** 4 / 4 - (-2) ** 2 + (-2)), 1e-10))

# --- endpoint singularities (the whole point of tanh-sinh) -----------------
check("integral of 1/sqrt(x) on [0,1] = 2 (singular at 0)",
      close(integrate(lambda x: 1 / math.sqrt(x) if x > 0 else 0.0, 0, 1), 2, 1e-6))
check("integral of ln(1/x) on [0,1] = 1 (log singularity at 0)",
      close(integrate(lambda x: -math.log(x) if x > 0 else 0.0, 0, 1), 1, 1e-9))
check("integral of 1/sqrt(1-x^2) on [0,1] = pi/2 (singular at 1)",
      close(integrate(lambda x: 1 / math.sqrt(1 - x * x) if x < 1 else 0.0, 0, 1), math.pi / 2, 1e-6))
check("integral of sqrt(x) on [0,1] = 2/3 (derivative singular at 0)",
      close(integrate(math.sqrt, 0, 1), 2 / 3, 1e-10))
check("integral of x^(-1/3) on [0,1] = 3/2",
      close(integrate(lambda x: x ** (-1 / 3) if x > 0 else 0.0, 0, 1), 1.5, 1e-5))

# --- Beta-function style: integral of x^(-1/2)(1-x)^(-1/2) = pi -------------
check("integral of 1/sqrt(x(1-x)) on [0,1] = pi (singular at both ends)",
      close(integrate(lambda x: 1 / math.sqrt(x * (1 - x)) if 0 < x < 1 else 0.0, 0, 1),
            math.pi, 1e-5))

# --- general interval [a,b] mapping ----------------------------------------
check("integral of x on [2,5] = 10.5", close(integrate(lambda x: x, 2, 5), 10.5, 1e-12))
check("integral of exp on [-1,2] = e^2 - e^-1",
      close(integrate(math.exp, -1, 2), math.exp(2) - math.exp(-1), 1e-11))
check("a == b gives 0", integrate(math.sin, 3.0, 3.0) == 0.0)

# --- linearity and additivity ----------------------------------------------
f = lambda x: math.cos(x) + x * x    # noqa: E731
whole = integrate(f, 0, 2)
part = integrate(f, 0, 1) + integrate(f, 1, 2)
check("additivity: integral over [0,2] equals [0,1]+[1,2]", close(whole, part, 1e-10))

# reversing limits negates (our map handles a<b; test via explicit negation)
check("integral of a Gaussian bump on [-3,3] ~ sqrt(pi) region",
      close(integrate(lambda x: math.exp(-x * x), -3, 3),
            # erf(3)*sqrt(pi) ; compute reference by fine Simpson
            _simpson_ref(lambda x: math.exp(-x * x), -3, 3, 200000), 1e-8))

# --- compare a smooth case to a fine reference -----------------------------
def _ref(f, a, b):
    return _simpson_ref(f, a, b, 200000)


ref_ok = True
for (f, a, b) in [(lambda x: math.sin(x) * math.exp(-x), 0, 5),
                  (lambda x: 1 / (1 + x ** 4), 0, 3),
                  (lambda x: math.cos(3 * x) ** 2, 0, math.pi)]:
    if not close(integrate(f, a, b), _ref(f, a, b), 1e-7):
        ref_ok = False
        break
check("smooth integrals match a fine Simpson reference", ref_ok)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all tanh_sinh tests passed")
