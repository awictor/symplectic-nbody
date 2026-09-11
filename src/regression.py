"""Linear and logistic regression: fitting a line, and a decision boundary.

The two workhorses of supervised learning. LINEAR regression fits y ~= w . x + b by minimizing
squared error; its closed form is the normal equations (solved here by QR least squares, which
avoids squaring the condition number), and it can also be reached by gradient descent -- useful
when the data is huge or streamed. LOGISTIC regression predicts a PROBABILITY p = sigmoid(w . x
+ b) for binary labels, fit by minimizing the log-loss (cross-entropy); there is no closed form,
so gradient descent (or Newton's method) is used, and the decision boundary w . x + b = 0 is a
hyperplane separating the classes.

Both are linear models -- the prediction is linear in the features -- but logistic squashes it
through the sigmoid to stay a probability, and its convex log-loss has a single global minimum,
so gradient descent always finds it. Regularization (an L2 penalty on the weights) trades a
little fit for smaller, more generalizable weights.

This module fits linear regression by both QR and gradient descent, logistic regression by
gradient descent with optional L2, and reports R^2 for regression and accuracy / log-loss for
classification, checked against exact fits, separable data, and the equivalence of the two
linear solvers. Pure stdlib; the supervised-learning companion to the k-means and SVD notes."""

from __future__ import annotations

import math

from qr import lstsq as _qr_lstsq


def _design(X):
    """Prepend a 1 to each feature row for the intercept term."""
    return [[1.0] + list(row) for row in X]


def _predict_linear(row_with_bias, w):
    return sum(a * b for a, b in zip(row_with_bias, w))


def linear_fit(X, y):
    """Ordinary least-squares linear regression via QR. X is a list of feature rows (no bias
    column -- one is added), y the targets. Returns the weight vector [bias, w1, w2, ...]."""
    A = _design(X)
    return _qr_lstsq(A, list(y))


def linear_fit_gd(X, y, lr: float = 0.01, epochs: int = 5000, l2: float = 0.0):
    """Linear regression by batch gradient descent (for large/streamed data or as a check on
    the closed form). Returns [bias, w1, ...]."""
    A = _design(X)
    n = len(A)
    d = len(A[0])
    w = [0.0] * d
    for _ in range(epochs):
        grad = [0.0] * d
        for i in range(n):
            err = _predict_linear(A[i], w) - y[i]
            for j in range(d):
                grad[j] += err * A[i][j]
        for j in range(d):
            reg = l2 * w[j] if j > 0 else 0.0        # do not regularize the bias
            w[j] -= lr * (grad[j] / n + reg)
    return w


def predict(X, w):
    """Linear predictions for feature rows X given weights [bias, ...]."""
    A = _design(X)
    return [_predict_linear(row, w) for row in A]


def r_squared(X, y, w) -> float:
    """Coefficient of determination R^2 = 1 - SS_res / SS_tot: the fraction of variance the fit
    explains (1 = perfect, 0 = no better than the mean)."""
    preds = predict(X, w)
    ybar = sum(y) / len(y)
    ss_res = sum((y[i] - preds[i]) ** 2 for i in range(len(y)))
    ss_tot = sum((yi - ybar) ** 2 for yi in y)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


def mse(X, y, w) -> float:
    """Mean squared error of the linear fit."""
    preds = predict(X, w)
    return sum((y[i] - preds[i]) ** 2 for i in range(len(y))) / len(y)


# --- logistic regression ----------------------------------------------------

def sigmoid(z: float) -> float:
    """The logistic function 1/(1+e^-z), numerically stable for large |z|."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def logistic_fit(X, y, lr: float = 0.1, epochs: int = 5000, l2: float = 0.0):
    """Logistic regression by gradient descent on the log-loss. y is 0/1 labels. Returns weights
    [bias, w1, ...]; predictions are sigmoid(w . [1, x])."""
    A = _design(X)
    n = len(A)
    d = len(A[0])
    w = [0.0] * d
    for _ in range(epochs):
        grad = [0.0] * d
        for i in range(n):
            p = sigmoid(_predict_linear(A[i], w))
            err = p - y[i]                            # gradient of log-loss is (p - y) x
            for j in range(d):
                grad[j] += err * A[i][j]
        for j in range(d):
            reg = l2 * w[j] if j > 0 else 0.0
            w[j] -= lr * (grad[j] / n + reg)
    return w


def predict_proba(X, w):
    """Predicted class-1 probabilities for logistic weights."""
    A = _design(X)
    return [sigmoid(_predict_linear(row, w)) for row in A]


def predict_class(X, w, threshold: float = 0.5):
    """Predicted 0/1 labels at the given probability threshold."""
    return [1 if p >= threshold else 0 for p in predict_proba(X, w)]


def accuracy(X, y, w) -> float:
    """Classification accuracy: fraction of labels predicted correctly."""
    preds = predict_class(X, w)
    return sum(1 for i in range(len(y)) if preds[i] == y[i]) / len(y)


def log_loss(X, y, w) -> float:
    """Mean binary cross-entropy -(y ln p + (1-y) ln(1-p)); the logistic objective."""
    probs = predict_proba(X, w)
    total = 0.0
    for i in range(len(y)):
        p = min(max(probs[i], 1e-15), 1 - 1e-15)     # clip to avoid log(0)
        total += -(y[i] * math.log(p) + (1 - y[i]) * math.log(1 - p))
    return total / len(y)
