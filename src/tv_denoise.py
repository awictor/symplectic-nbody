"""Total-variation denoising: removing noise while keeping edges sharp, exactly, in 1-D.

Ordinary smoothing (a moving average, a Gaussian blur) removes noise but also BLURS EDGES -- a sharp
step becomes a gentle ramp. TOTAL-VARIATION denoising, the 1-D fused lasso / Rudin-Osher-Fatemi
model, does better: it finds the signal u minimizing

    (1/2) sum (u_i - y_i)^2  +  lambda * sum |u_{i+1} - u_i|,

trading fidelity to the noisy data y against the TOTAL VARIATION (sum of absolute jumps). The L1
penalty on jumps is the magic: unlike an L2 penalty it drives most differences to EXACTLY zero,
producing a piecewise-CONSTANT result that keeps genuine edges crisp while flattening noise into
plateaus. It is the workhorse behind edge-preserving image denoising, changepoint detection, and
genomic copy-number segmentation.

The 1-D problem is convex, and its exact minimizer is found here by DUAL PROJECTED-GRADIENT descent:
substituting u = y - D^T p turns the problem into maximizing a smooth concave dual over the box
|p_i| <= lambda (p is a variable on each of the n-1 differences), which gradient ascent with a simple
per-step clip to the box solves. Iterating to convergence gives the exact denoised signal; because the
dual is smooth and box-constrained, it converges reliably. This module runs it, exposes the TV
functional and a lambda-sweep helper, and ships an independent brute-force check on tiny signals.

Validated against a fine-grid brute-force optimum on small signals and by the model's properties: the
denoiser recovers a clean piecewise-constant signal from noise; larger lambda gives fewer, longer
plateaus and lower total variation; lambda = 0 returns the input; a huge lambda collapses to the data
mean; the output stays within the data range (no overshoot); and a sharp step is preserved. Pure
stdlib; the edge-preserving companion to the Savitzky-Golay / kernel smoothers and the
isotonic-regression shape-constrained fit."""

from __future__ import annotations


def tv_denoise(y, lam, iterations=20000, step=0.2):
    """1-D total-variation denoising: minimize 0.5*sum(u-y)^2 + lam*sum|u_{i+1}-u_i| by dual
    projected-gradient descent (u = y - D^T p, |p| <= lam). Returns the denoised signal u."""
    n = len(y)
    if n == 0:
        return []
    if lam <= 0 or n == 1:
        return list(y)
    p = [0.0] * (n - 1)

    def primal(pp):
        u = list(y)
        for i in range(n - 1):
            u[i] -= pp[i]
            u[i + 1] += pp[i]
        return u

    # step must be < 1/||D||^2_op; ||D||^2 <= 4, so step <= 0.25 is safe. Clamp the caller's value.
    step = min(step, 0.24)
    for _ in range(iterations):
        u = primal(p)
        max_change = 0.0
        for i in range(n - 1):
            old = p[i]
            p[i] += step * (u[i] - u[i + 1])
            if p[i] > lam:
                p[i] = lam
            elif p[i] < -lam:
                p[i] = -lam
            max_change = max(max_change, abs(p[i] - old))
        if max_change < 1e-11:
            break
    return primal(p)


def total_variation(u):
    """The total variation sum |u_{i+1} - u_i|."""
    return sum(abs(u[i + 1] - u[i]) for i in range(len(u) - 1))


def objective(u, y, lam):
    """The denoising objective 0.5*sum(u-y)^2 + lam*TV(u)."""
    fidelity = 0.5 * sum((u[i] - y[i]) ** 2 for i in range(len(y)))
    return fidelity + lam * total_variation(u)


def n_plateaus(u, tol=1e-7):
    """Number of constant segments (plateaus) in u."""
    if not u:
        return 0
    count = 1
    for i in range(1, len(u)):
        if abs(u[i] - u[i - 1]) > tol:
            count += 1
    return count


def lambda_path(y, lambdas):
    """Denoise y at each lambda; returns a list of (lambda, TV, n_plateaus, objective)."""
    out = []
    for lam in lambdas:
        u = tv_denoise(y, lam)
        out.append((lam, total_variation(u), n_plateaus(u), objective(u, y, lam)))
    return out


# --- reference: fine-grid brute-force optimum (tiny signals only) ------------
def brute_tv_optimum(y, lam, grid_lo=None, grid_hi=None, steps=60):
    """Independent check: minimize the TV objective by exhaustive search on a fine value grid.
    Only tractable for signals of length <= 3 or 4. Returns (best_u, best_objective)."""
    import itertools
    n = len(y)
    if grid_lo is None:
        grid_lo = min(y) - 1
    if grid_hi is None:
        grid_hi = max(y) + 1
    grid = [grid_lo + (grid_hi - grid_lo) * k / steps for k in range(steps + 1)]
    best = float("inf")
    best_u = None
    for combo in itertools.product(grid, repeat=n):
        o = objective(list(combo), y, lam)
        if o < best:
            best = o
            best_u = list(combo)
    return best_u, best
