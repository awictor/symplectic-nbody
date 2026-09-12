"""Tests for autodiff: gradients vs finite differences and symbolic, diamonds, descent, chain rule."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from autodiff import Value, grad, finite_diff_grad

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 808
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


# --- a suite of functions: AD gradient matches finite differences ----------
suite = [
    # (autodiff fn on Values, plain fn on floats, n_inputs)
    (lambda v: v[0] * v[1] + v[0], lambda v: v[0] * v[1] + v[0], 2),
    (lambda v: v[0] ** 3 - 2 * v[0] ** 2 + 1, lambda v: v[0] ** 3 - 2 * v[0] ** 2 + 1, 1),
    (lambda v: (v[0] * v[1]).sin() + v[2].exp(), lambda v: math.sin(v[0] * v[1]) + math.exp(v[2]), 3),
    (lambda v: (v[0] ** 2 + v[1] ** 2).sqrt(), lambda v: math.sqrt(v[0] ** 2 + v[1] ** 2), 2),
    (lambda v: v[0] / v[1] + v[1].log(), lambda v: v[0] / v[1] + math.log(v[1]), 2),
    (lambda v: (v[0].tanh() * v[1]).cos(), lambda v: math.cos(math.tanh(v[0]) * v[1]), 2),
    (lambda v: (2 * v[0]).relu() + v[1] ** 4, lambda v: max(0.0, 2 * v[0]) + v[1] ** 4, 2),
]

all_ok = True
for ad_fn, pl_fn, n in suite:
    for _ in range(20):
        xs = [rng() * 2 + 0.3 for _ in range(n)]     # positive so log/sqrt are defined
        _, g_ad = grad(ad_fn, xs)
        g_fd = finite_diff_grad(pl_fn, xs)
        if any(abs(g_ad[i] - g_fd[i]) > 1e-4 for i in range(n)):
            all_ok = False
            break
    if not all_ok:
        break
check("autodiff gradients match finite differences across a function suite", all_ok)

# --- symbolic checks -------------------------------------------------------
# f(x,y) = x^2 y + sin x + e^y ; df/dx = 2xy + cos x, df/dy = x^2 + e^y
def f(v):
    x, y = v
    return x * x * y + x.sin() + y.exp()

x0, y0 = 1.3, 0.6
_, g = grad(f, [x0, y0])
check("d/dx matches symbolic 2xy + cos x", abs(g[0] - (2 * x0 * y0 + math.cos(x0))) < 1e-9)
check("d/dy matches symbolic x^2 + e^y", abs(g[1] - (x0 * x0 + math.exp(y0))) < 1e-9)

# --- diamond: a value used multiple times accumulates gradients ------------
# z = x*x -> dz/dx = 2x
check("diamond x*x has gradient 2x", abs(grad(lambda v: v[0] * v[0], [4.0])[1][0] - 8.0) < 1e-12)
# z = x*x*x -> 3x^2
check("cubic x*x*x has gradient 3x^2", abs(grad(lambda v: v[0] * v[0] * v[0], [2.0])[1][0] - 12.0) < 1e-12)
# z = (x+x)*(x+x) = 4x^2 -> 8x
check("reused sum (x+x)^2 accumulates", abs(grad(lambda v: (v[0] + v[0]) * (v[0] + v[0]), [1.5])[1][0] - 12.0) < 1e-12)

# --- deep chain rule -------------------------------------------------------
# sin(sin(sin(x))) ; derivative by chain rule
def deep(v):
    x = v[0]
    return x.sin().sin().sin()

x0 = 0.5
_, g = grad(deep, [x0])
sym = math.cos(math.sin(math.sin(x0))) * math.cos(math.sin(x0)) * math.cos(x0)
check("deep chain rule sin(sin(sin(x)))", abs(g[0] - sym) < 1e-9)

# --- gradient descent using autodiff reaches the analytic optimum ----------
# minimize f(x,y) = (x-3)^2 + (y+1)^2  -> optimum (3, -1)
def bowl(v):
    return (v[0] - 3) ** 2 + (v[1] + 1) ** 2

xs = [0.0, 0.0]
for _ in range(200):
    _, g = grad(bowl, xs)
    xs = [xs[i] - 0.1 * g[i] for i in range(2)]
check("gradient descent via autodiff finds the optimum (3,-1)",
      abs(xs[0] - 3) < 1e-3 and abs(xs[1] + 1) < 1e-3)

# --- Rosenbrock gradient matches the analytic formula ----------------------
# f = 100(y - x^2)^2 + (1 - x)^2
def rosen(v):
    x, y = v
    return 100 * (y - x * x) ** 2 + (1 - x) ** 2

x0, y0 = 0.5, 0.7
_, g = grad(rosen, [x0, y0])
gx = -400 * x0 * (y0 - x0 * x0) - 2 * (1 - x0)
gy = 200 * (y0 - x0 * x0)
check("Rosenbrock gradient matches analytic", abs(g[0] - gx) < 1e-6 and abs(g[1] - gy) < 1e-6)

# --- operator coverage: r-operators and division ---------------------------
check("2 - x gradient is -1", abs(grad(lambda v: 2 - v[0], [5.0])[1][0] + 1) < 1e-12)
check("1 / x gradient is -1/x^2", abs(grad(lambda v: 1 / v[0], [2.0])[1][0] + 0.25) < 1e-12)
check("x / y gradients", (lambda r: abs(r[1][0] - 0.5) < 1e-9 and abs(r[1][1] + 0.75) < 1e-9)(
    grad(lambda v: v[0] / v[1], [3.0, 2.0])))

# --- forward value is correct ----------------------------------------------
val, _ = grad(lambda v: v[0] * v[1] + v[0].exp(), [2.0, 3.0])
check("forward value is correct", abs(val - (6.0 + math.exp(2.0))) < 1e-9)

# --- backward resets grads (re-callable) -----------------------------------
x = Value(2.0)
y = x * x
y.backward()
g1 = x.grad
y2 = x * x * x
y2.backward()
check("backward recomputes cleanly (3x^2 = 12)", abs(x.grad - 12.0) < 1e-12)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all autodiff tests passed")
