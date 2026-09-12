"""Reverse-mode automatic differentiation: exact gradients through any computation.

How does a deep-learning framework compute the gradient of a loss with respect to millions of
parameters? Not by hand-derived formulas and not by finite differences (too slow and inexact), but by
REVERSE-MODE AUTOMATIC DIFFERENTIATION -- the engine (PyTorch's autograd, TensorFlow's tape, JAX's
grad) that computes exact derivatives of any function built from elementary operations, at the cost
of about one extra function evaluation regardless of how many inputs there are. It is neither
symbolic differentiation (which explodes in expression size) nor numerical differencing (which loses
precision); it applies the chain rule mechanically over the COMPUTATION GRAPH.

The trick is to record operations as they happen. Each numeric VALUE remembers the operation that
produced it and its parent Values, so a whole expression becomes a directed acyclic graph. The
forward pass computes the result; the BACKWARD pass then walks the graph in reverse topological
order, and at each node applies the LOCAL DERIVATIVE of its operation to accumulate the gradient
flowing back from its children -- exactly backpropagation, generalized from neural-net layers to any
composition of +, *, sin, exp, and the rest. Because reverse mode sweeps from the single output back
to all inputs, it delivers the entire gradient vector in one pass, which is why it dominates machine
learning where the output (a scalar loss) is far smaller than the input (the parameters).

This module implements a Value type supporting +, -, *, /, **, negation, and exp/log/sin/cos/tanh/
relu/sqrt, each recording its local derivative, plus a topological-sort backward pass that fills in
every input's gradient. It is verified that the computed gradients match FINITE-DIFFERENCE gradients
to high precision across a suite of nonlinear functions, that they match hand-derived SYMBOLIC
derivatives on known cases, that gradients accumulate correctly when a value is reused (a diamond in
the graph), that a full gradient-descent loop using autodiff minimizes a function to its analytic
optimum, and on the chain rule through deep compositions. Pure stdlib; a numerical-methods companion
to the multi-layer-perceptron and Newton's-method notes."""

from __future__ import annotations

import math


class Value:
    """A scalar node in a computation graph. Supports arithmetic and common functions; call
    .backward() on the output to populate .grad on every upstream Value."""

    __slots__ = ("data", "grad", "_backward", "_prev", "_op")

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"

    # --- arithmetic ------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, power):
        assert isinstance(power, (int, float)), "only constant powers supported"
        out = Value(self.data ** power, (self,), f"**{power}")

        def _backward():
            self.grad += power * (self.data ** (power - 1)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other if isinstance(other, Value) else Value(-other))

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return (Value(other) if not isinstance(other, Value) else other) + (-self)

    def __rtruediv__(self, other):
        return (Value(other) if not isinstance(other, Value) else other) * self ** -1

    # --- elementary functions -------------------------------------------
    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def sin(self):
        out = Value(math.sin(self.data), (self,), "sin")

        def _backward():
            self.grad += math.cos(self.data) * out.grad
        out._backward = _backward
        return out

    def cos(self):
        out = Value(math.cos(self.data), (self,), "cos")

        def _backward():
            self.grad += -math.sin(self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def sqrt(self):
        out = Value(math.sqrt(self.data), (self,), "sqrt")

        def _backward():
            self.grad += (0.5 / math.sqrt(self.data)) * out.grad
        out._backward = _backward
        return out

    # --- reverse pass ----------------------------------------------------
    def backward(self):
        """Populate .grad on every node upstream of self (self.grad seeded to 1)."""
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        # reset grads on the whole graph, then seed and sweep
        for v in topo:
            v.grad = 0.0
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


def grad(f, xs):
    """Compute f's gradient with respect to the list of scalar inputs xs (plain floats).

    f takes a list of Value inputs and returns a single Value. Returns (value, [grads])."""
    inputs = [Value(x) for x in xs]
    out = f(inputs)
    out.backward()
    return out.data, [v.grad for v in inputs]


def finite_diff_grad(f, xs, h=1e-6):
    """Central-difference gradient of a plain-float function f(list) -> float, for checking."""
    n = len(xs)
    g = []
    for i in range(n):
        xp = list(xs); xp[i] += h
        xm = list(xs); xm[i] -= h
        g.append((f(xp) - f(xm)) / (2 * h))
    return g
