"""Multi-layer perceptron and backpropagation: learning nonlinear functions from scratch.

A single linear classifier cannot learn XOR -- no straight line separates its two classes. The
MULTI-LAYER PERCEPTRON (MLP) overcomes this by stacking layers of neurons with NONLINEAR activation
functions: each layer computes a weighted sum of its inputs, passes it through a squashing
nonlinearity, and feeds the result forward. With even one hidden layer an MLP is a UNIVERSAL
APPROXIMATOR -- it can represent any continuous function to arbitrary accuracy -- and it is the
foundation of modern deep learning.

The MLP learns by BACKPROPAGATION, the algorithm that made neural networks practical (Rumelhart,
Hinton, Williams, 1986). It is the chain rule applied systematically: run an input forward through
the network to get a prediction and a loss; then propagate the loss's gradient BACKWARD layer by
layer, each layer computing how much its weights contributed to the error from the gradient handed
back by the layer above. Because each layer reuses the gradient of the layer downstream, the whole
gradient is computed in one backward pass costing the same as the forward pass -- the efficiency that
makes training deep networks feasible. Gradient descent then nudges every weight down its gradient,
and repeating over the data drives the loss down until the network fits.

This module implements a feedforward MLP with configurable layer sizes and activations (sigmoid,
tanh, ReLU), full backpropagation, and mini-batch gradient descent with momentum, using a seeded RNG
so training is reproducible. It is verified that its analytic gradients match FINITE-DIFFERENCE
gradients (the definitive backprop-correctness test), that it learns the XOR function that defeats a
linear model (driving the loss near zero and classifying all four points correctly), that it fits a
noisy nonlinear regression target, that it separates two interleaving spiral/blob classes to high
accuracy, and that training is reproducible from a seed. Pure stdlib; a machine-learning companion to
the logistic-regression, k-means, and decision-tree notes."""

from __future__ import annotations

import math


def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


_ACTIVATIONS = {
    "sigmoid": (lambda x: _sigmoid(x), lambda y: y * (1 - y)),          # derivative in terms of output
    "tanh": (lambda x: math.tanh(x), lambda y: 1 - y * y),
    "relu": (lambda x: x if x > 0 else 0.0, lambda y: 1.0 if y > 0 else 0.0),
    "identity": (lambda x: x, lambda y: 1.0),
}


class MLP:
    """A feedforward multi-layer perceptron trained by backpropagation.

    layers: list of layer sizes [n_in, h1, h2, ..., n_out]. activation: hidden-layer activation.
    output_activation: usually 'sigmoid' for classification, 'identity' for regression."""

    def __init__(self, layers, activation="tanh", output_activation="sigmoid", seed=1):
        self.layers = layers
        self.act_name = activation
        self.out_name = output_activation
        self.act, self.act_d = _ACTIVATIONS[activation]
        self.out_act, self.out_act_d = _ACTIVATIONS[output_activation]
        self._state = seed & 0xFFFFFFFF
        # weights[l][j][i] = weight from unit i of layer l to unit j of layer l+1; biases[l][j]
        self.weights = []
        self.biases = []
        for l in range(len(layers) - 1):
            n_in, n_out = layers[l], layers[l + 1]
            # Xavier-ish init scaled by fan-in
            scale = (1.0 / n_in) ** 0.5
            w = [[self._randn() * scale for _ in range(n_in)] for _ in range(n_out)]
            b = [0.0 for _ in range(n_out)]
            self.weights.append(w)
            self.biases.append(b)

    def _rand(self):
        self._state = (1664525 * self._state + 1013904223) & 0xFFFFFFFF
        return (self._state >> 8) / (1 << 24)

    def _randn(self):
        # Box-Muller
        u1 = max(self._rand(), 1e-12)
        u2 = self._rand()
        return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

    def forward(self, x):
        """Forward pass. Returns the list of per-layer activations (activations[0] == input)."""
        activations = [list(x)]
        n_layers = len(self.weights)
        for l in range(n_layers):
            act_fn = self.out_act if l == n_layers - 1 else self.act
            prev = activations[-1]
            layer_out = []
            for j in range(self.layers[l + 1]):
                z = self.biases[l][j] + sum(self.weights[l][j][i] * prev[i] for i in range(self.layers[l]))
                layer_out.append(act_fn(z))
            activations.append(layer_out)
        return activations

    def predict(self, x):
        return self.forward(x)[-1]

    def _backprop(self, x, y):
        """One sample's gradients. Returns (grad_w, grad_b) matching the shapes of weights/biases and
        the sample loss (squared error)."""
        activations = self.forward(x)
        n_layers = len(self.weights)
        # output-layer delta (squared-error loss): (out - y) * out'(z)
        out = activations[-1]
        deltas = [None] * n_layers
        delta = [(out[j] - y[j]) * self.out_act_d(out[j]) for j in range(len(out))]
        deltas[-1] = delta
        # backpropagate
        for l in range(n_layers - 2, -1, -1):
            a = activations[l + 1]
            nxt_delta = deltas[l + 1]
            layer_delta = []
            for j in range(self.layers[l + 1]):
                err = sum(self.weights[l + 1][k][j] * nxt_delta[k] for k in range(self.layers[l + 2]))
                layer_delta.append(err * self.act_d(a[j]))
            deltas[l] = layer_delta
        # gradients
        grad_w = []
        grad_b = []
        for l in range(n_layers):
            prev = activations[l]
            gw = [[deltas[l][j] * prev[i] for i in range(self.layers[l])]
                  for j in range(self.layers[l + 1])]
            gb = list(deltas[l])
            grad_w.append(gw)
            grad_b.append(gb)
        loss = 0.5 * sum((out[j] - y[j]) ** 2 for j in range(len(out)))
        return grad_w, grad_b, loss

    def train(self, X, Y, epochs=1000, lr=0.1, momentum=0.9, batch_size=None):
        """Train by mini-batch gradient descent with momentum. Returns the loss history (per epoch)."""
        n = len(X)
        batch_size = batch_size or n
        vw = [[[0.0] * self.layers[l] for _ in range(self.layers[l + 1])] for l in range(len(self.weights))]
        vb = [[0.0] * self.layers[l + 1] for l in range(len(self.weights))]
        history = []
        for epoch in range(epochs):
            # simple shuffle via the RNG
            idx = list(range(n))
            for i in range(n - 1, 0, -1):
                j = int(self._rand() * (i + 1))
                idx[i], idx[j] = idx[j], idx[i]
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                batch = idx[start:start + batch_size]
                # accumulate gradients over the batch
                acc_w = [[[0.0] * self.layers[l] for _ in range(self.layers[l + 1])]
                         for l in range(len(self.weights))]
                acc_b = [[0.0] * self.layers[l + 1] for l in range(len(self.weights))]
                for bi in batch:
                    gw, gb, loss = self._backprop(X[bi], Y[bi])
                    epoch_loss += loss
                    for l in range(len(self.weights)):
                        for j in range(self.layers[l + 1]):
                            acc_b[l][j] += gb[l][j]
                            for i in range(self.layers[l]):
                                acc_w[l][j][i] += gw[l][j][i]
                m = len(batch)
                for l in range(len(self.weights)):
                    for j in range(self.layers[l + 1]):
                        vb[l][j] = momentum * vb[l][j] - lr * acc_b[l][j] / m
                        self.biases[l][j] += vb[l][j]
                        for i in range(self.layers[l]):
                            vw[l][j][i] = momentum * vw[l][j][i] - lr * acc_w[l][j][i] / m
                            self.weights[l][j][i] += vw[l][j][i]
            history.append(epoch_loss / n)
        return history

    def flat_params(self):
        """All weights and biases flattened into one list (for finite-difference gradient checks)."""
        out = []
        for l in range(len(self.weights)):
            for j in range(self.layers[l + 1]):
                out.extend(self.weights[l][j])
                out.append(self.biases[l][j])
        return out

    def set_flat_params(self, flat):
        k = 0
        for l in range(len(self.weights)):
            for j in range(self.layers[l + 1]):
                for i in range(self.layers[l]):
                    self.weights[l][j][i] = flat[k]; k += 1
                self.biases[l][j] = flat[k]; k += 1
