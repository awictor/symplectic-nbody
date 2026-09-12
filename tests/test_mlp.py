"""Tests for mlp: gradient check vs finite differences, XOR, regression, classification, seed."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlp import MLP

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 5150
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


# --- gradient check: analytic backprop vs finite differences ---------------
def loss_on(net, X, Y):
    total = 0.0
    for x, y in zip(X, Y):
        out = net.predict(x)
        total += 0.5 * sum((out[j] - y[j]) ** 2 for j in range(len(y)))
    return total


net = MLP([2, 3, 2], activation="tanh", output_activation="sigmoid", seed=7)
X = [[0.5, -0.3], [0.1, 0.8]]
Y = [[1, 0], [0, 1]]

# analytic gradient summed over the samples
n_layers = len(net.weights)
acc_w = [[[0.0] * net.layers[l] for _ in range(net.layers[l + 1])] for l in range(n_layers)]
acc_b = [[0.0] * net.layers[l + 1] for l in range(n_layers)]
for x, y in zip(X, Y):
    gw, gb, _ = net._backprop(x, y)
    for l in range(n_layers):
        for j in range(net.layers[l + 1]):
            acc_b[l][j] += gb[l][j]
            for i in range(net.layers[l]):
                acc_w[l][j][i] += gw[l][j][i]

# finite-difference gradient over the flattened params
flat = net.flat_params()
eps = 1e-6
max_err = 0.0
# map flat index back to (kind); easier: perturb each and compare to the flattened analytic grad
analytic_flat = []
for l in range(n_layers):
    for j in range(net.layers[l + 1]):
        analytic_flat.extend(acc_w[l][j])
        analytic_flat.append(acc_b[l][j])

for k in range(len(flat)):
    saved = flat[k]
    flat[k] = saved + eps
    net.set_flat_params(flat)
    lp = loss_on(net, X, Y)
    flat[k] = saved - eps
    net.set_flat_params(flat)
    lm = loss_on(net, X, Y)
    flat[k] = saved
    net.set_flat_params(flat)
    fd = (lp - lm) / (2 * eps)
    max_err = max(max_err, abs(fd - analytic_flat[k]))
check(f"backprop gradients match finite differences (max err {max_err:.2e})", max_err < 1e-6)

# --- XOR: the classic nonlinear problem ------------------------------------
Xxor = [[0, 0], [0, 1], [1, 0], [1, 1]]
Yxor = [[0], [1], [1], [0]]
net = MLP([2, 4, 1], activation="tanh", output_activation="sigmoid", seed=3)
hist = net.train(Xxor, Yxor, epochs=3000, lr=0.5)
check("XOR loss decreases far", hist[-1] < hist[0] * 0.05)
check("XOR final loss is near zero", hist[-1] < 0.01)
preds = [net.predict(x)[0] for x in Xxor]
check("XOR classifies all four points correctly",
      all((preds[i] > 0.5) == bool(Yxor[i][0]) for i in range(4)))

# --- a linear model cannot solve XOR (contrast) ----------------------------
lin = MLP([2, 1], activation="identity", output_activation="sigmoid", seed=1)
lin.train(Xxor, Yxor, epochs=2000, lr=0.5)
lin_preds = [lin.predict(x)[0] for x in Xxor]
lin_correct = sum((lin_preds[i] > 0.5) == bool(Yxor[i][0]) for i in range(4))
check("a linear model fails to solve XOR (<= 3 of 4)", lin_correct <= 3)

# --- nonlinear regression: fit y = sin(pi x) on [0,1] ----------------------
Xr = [[i / 40] for i in range(41)]
Yr = [[0.5 + 0.5 * math.sin(math.pi * (2 * (i / 40) - 1))] for i in range(41)]   # scaled to [0,1]
reg = MLP([1, 8, 1], activation="tanh", output_activation="sigmoid", seed=2)
hist = reg.train(Xr, Yr, epochs=3000, lr=0.3)
mse = sum((reg.predict(Xr[i])[0] - Yr[i][0]) ** 2 for i in range(41)) / 41
check(f"nonlinear regression fits with small MSE ({mse:.4f})", mse < 0.01)

# --- classification: two separable blobs -----------------------------------
def make_blobs(n):
    X, Y = [], []
    for _ in range(n):
        if rng() < 0.5:
            X.append([rng() * 2 - 3, rng() * 2 - 3]); Y.append([0])   # lower-left
        else:
            X.append([rng() * 2 + 1, rng() * 2 + 1]); Y.append([1])   # upper-right
    return X, Y


Xb, Yb = make_blobs(200)
clf = MLP([2, 6, 1], activation="tanh", output_activation="sigmoid", seed=4)
clf.train(Xb, Yb, epochs=500, lr=0.2)
acc = sum((clf.predict(Xb[i])[0] > 0.5) == bool(Yb[i][0]) for i in range(200)) / 200
check(f"two-blob classification accuracy is high ({acc:.2f})", acc > 0.95)

# --- reproducibility -------------------------------------------------------
a = MLP([2, 3, 1], seed=11)
b = MLP([2, 3, 1], seed=11)
ha = a.train(Xxor, Yxor, epochs=100, lr=0.3)
hb = b.train(Xxor, Yxor, epochs=100, lr=0.3)
check("same seed gives identical training", ha == hb)

# --- ReLU activation also learns XOR ---------------------------------------
relu_net = MLP([2, 8, 1], activation="relu", output_activation="sigmoid", seed=6)
relu_hist = relu_net.train(Xxor, Yxor, epochs=3000, lr=0.1)
relu_preds = [relu_net.predict(x)[0] for x in Xxor]
relu_correct = sum((relu_preds[i] > 0.5) == bool(Yxor[i][0]) for i in range(4))
check(f"ReLU network also learns XOR ({relu_correct}/4 correct)", relu_correct == 4)

# --- output shape ----------------------------------------------------------
multi = MLP([3, 5, 4], seed=1)
out = multi.predict([0.1, 0.2, 0.3])
check("multi-output network returns the right shape", len(out) == 4)
check("sigmoid outputs are in (0,1)", all(0 < v < 1 for v in out))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all mlp tests passed")
