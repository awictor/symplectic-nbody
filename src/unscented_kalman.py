"""The unscented Kalman filter -- state estimation when the dynamics are nonlinear.

The ordinary Kalman filter is optimal, but only for LINEAR systems: it assumes the state evolves and is
measured by matrix multiplication. Reality rarely obliges. A radar tracks a target in range and bearing
but wants position and velocity in Cartesian coordinates -- a nonlinear map. A pendulum's angle obeys a
sine. A chemical reactor, a spacecraft re-entry, a robot's pose from wheel odometry: all nonlinear. The
classic fix, the EXTENDED Kalman filter, linearises by taking Jacobians -- derivatives of the dynamics --
which is fiddly to derive, fragile when the functions are sharply curved, and simply wrong when they are
not differentiable. The UNSCENTED Kalman filter (Julier & Uhlmann, 1997) sidesteps all of that with a
better idea: it is easier to approximate a probability distribution than an arbitrary nonlinear function.

The engine is the UNSCENTED TRANSFORM. Rather than linearise, deterministically pick a small set of
SIGMA POINTS -- the mean, plus points spread one "square root of the covariance" away along each
dimension -- that exactly capture the mean and covariance of the current state estimate. Push every
sigma point through the true nonlinear function, untouched, and then recompute a weighted mean and
covariance of the transformed cloud. That recovered mean and covariance are accurate to SECOND order for
any nonlinearity (third for Gaussians), with no derivatives ever taken. The spread and weighting are
governed by three parameters -- alpha (how far to spread), beta (2 is optimal for Gaussians), and kappa
(a secondary scaling) -- combined into a scaling lambda = alpha^2 (n + kappa) - n.

A UKF cycle is the same predict/update rhythm as the linear filter, but every place the linear filter
multiplied by a matrix, the UKF instead sends sigma points through a function:

  * PREDICT: build sigma points from the current (mean, covariance), pass each through the process model
    f, and recover the predicted mean and covariance, adding the process noise Q.
  * UPDATE: pass the predicted sigma points through the measurement model h, recover the predicted
    measurement mean and its covariance (plus measurement noise R), form the state-measurement
    cross-covariance, compute the Kalman gain from those, and correct the state by the innovation.

The matrix square root of the covariance (needed to place the sigma points) is done here by an exact
Cholesky factorisation. Everything is pure standard library -- small dense linear algebra written out by
hand.

Validation. (1) On a genuinely LINEAR system the UKF must reproduce the ordinary Kalman filter's
estimates to numerical precision -- because the unscented transform is exact for linear (affine)
functions -- and this is checked step by step against the repository's own linear KalmanFilter. (2) The
recovered covariance stays symmetric positive-definite throughout. (3) On nonlinear tracking problems --
a projectile observed only in range/bearing, a pendulum observed only by angle -- the filter's RMS error
against the true hidden state is small and far below the raw measurement noise, and the estimate stays
inside its own reported uncertainty band about the right fraction of the time (consistency). (4) With
zero process and measurement noise on a linear identity system it is exact. (5) The sigma-point weights
sum to one and the transform recovers the mean and covariance of an untransformed Gaussian exactly."""

import math


# ---------------------------------------------------------------------------
# tiny dense linear algebra
# ---------------------------------------------------------------------------

def _matvec(A, v):
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def _matmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]


def _transpose(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def _add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _cholesky(A):
    """Lower-triangular Cholesky factor L with L L^T = A, for symmetric positive-definite A."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                d = A[i][i] - s
                if d <= 0:
                    d = 1e-12          # nudge for numerical safety
                L[i][j] = math.sqrt(d)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L


def _inverse(A):
    """Inverse of a small square matrix by Gauss-Jordan elimination."""
    n = len(A)
    M = [list(A[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        pivot = M[col][col]
        M[col] = [x / pivot for x in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0:
                factor = M[r][col]
                M[r] = [M[r][k] - factor * M[col][k] for k in range(2 * n)]
    return [row[n:] for row in M]


# ---------------------------------------------------------------------------
# sigma points and the unscented transform
# ---------------------------------------------------------------------------

class UnscentedKalmanFilter:
    """Unscented Kalman filter for a state of dimension n with measurements of dimension m.

    ``fx(state, dt)`` is the process model, ``hx(state)`` the measurement model. ``alpha``, ``beta``,
    ``kappa`` control the sigma-point spread (defaults are the standard textbook choice).
    """

    def __init__(self, n, m, fx, hx, alpha=1e-3, beta=2.0, kappa=0.0):
        self.n = n
        self.m = m
        self.fx = fx
        self.hx = hx
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lam = alpha * alpha * (n + kappa) - n

        # weights for mean (Wm) and covariance (Wc)
        self.Wm = [self.lam / (n + self.lam)]
        self.Wc = [self.lam / (n + self.lam) + (1 - alpha * alpha + beta)]
        for _ in range(2 * n):
            w = 1.0 / (2 * (n + self.lam))
            self.Wm.append(w)
            self.Wc.append(w)

        self.x = [0.0] * n                                   # state mean
        self.P = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]  # covariance

    def _sigma_points(self, x, P):
        n = self.n
        c = n + self.lam
        L = _cholesky([[c * P[i][j] for j in range(n)] for i in range(n)])
        pts = [list(x)]
        for k in range(n):
            col = [L[i][k] for i in range(n)]
            pts.append([x[i] + col[i] for i in range(n)])
        for k in range(n):
            col = [L[i][k] for i in range(n)]
            pts.append([x[i] - col[i] for i in range(n)])
        return pts

    def _unscented_mean_cov(self, points, noise, dim):
        mean = [0.0] * dim
        for w, p in zip(self.Wm, points):
            for i in range(dim):
                mean[i] += w * p[i]
        cov = [[0.0] * dim for _ in range(dim)]
        for w, p in zip(self.Wc, points):
            d = [p[i] - mean[i] for i in range(dim)]
            for i in range(dim):
                for j in range(dim):
                    cov[i][j] += w * d[i] * d[j]
        if noise is not None:
            cov = _add(cov, noise)
        return mean, cov

    def predict(self, dt, Q):
        """Propagate the state through the process model over step dt, adding process noise Q."""
        sigmas = self._sigma_points(self.x, self.P)
        self._pred_sigmas = [self.fx(s, dt) for s in sigmas]
        self.x, self.P = self._unscented_mean_cov(self._pred_sigmas, Q, self.n)

    def update(self, z, R):
        """Correct the state with a measurement z (length m) of noise covariance R."""
        # transform predicted sigma points through the measurement model
        zsig = [self.hx(s) for s in self._pred_sigmas]
        zmean, S = self._unscented_mean_cov(zsig, R, self.m)

        # cross-covariance between state and measurement
        Pxz = [[0.0] * self.m for _ in range(self.n)]
        for w, sp, zp in zip(self.Wc, self._pred_sigmas, zsig):
            dx = [sp[i] - self.x[i] for i in range(self.n)]
            dz = [zp[i] - zmean[i] for i in range(self.m)]
            for i in range(self.n):
                for j in range(self.m):
                    Pxz[i][j] += w * dx[i] * dz[j]

        K = _matmul(Pxz, _inverse(S))                        # Kalman gain
        innovation = [z[i] - zmean[i] for i in range(self.m)]
        self.x = [self.x[i] + sum(K[i][j] * innovation[j] for j in range(self.m))
                  for i in range(self.n)]
        # P = P - K S K^T
        KS = _matmul(K, S)
        KSKt = _matmul(KS, _transpose(K))
        self.P = [[self.P[i][j] - KSKt[i][j] for j in range(self.n)] for i in range(self.n)]

    # convenience accessors
    def state(self):
        return list(self.x)

    def covariance(self):
        return [row[:] for row in self.P]


def unscented_transform_stats(mean, cov, func, out_dim, alpha=1e-3, beta=2.0, kappa=0.0):
    """Standalone unscented transform: push a Gaussian (mean, cov) through ``func`` and return the
    transformed (mean, cov). Useful for testing the transform in isolation."""
    n = len(mean)
    ukf = UnscentedKalmanFilter(n, out_dim, lambda s, dt: s, func, alpha, beta, kappa)
    sig = ukf._sigma_points(mean, cov)
    transformed = [func(s) for s in sig]
    return ukf._unscented_mean_cov(transformed, None, out_dim)
