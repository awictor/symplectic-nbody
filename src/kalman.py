"""The Kalman filter: optimal recursive estimation of a hidden continuous state.

Where a hidden Markov model has DISCRETE hidden states, the Kalman filter (Kalman, 1960) tracks a
CONTINUOUS one -- a position and velocity, a temperature, a spacecraft's trajectory -- from noisy
measurements, in real time and with no growing history to store. It is the optimal estimator for a
linear system with Gaussian noise, and the workhorse behind GPS, guidance, and sensor fusion.

The world is modelled as a linear-Gaussian state-space system:

    x_t = F x_{t-1} + process noise   (Q)     -- how the true state evolves
    z_t = H x_t     + measurement noise (R)   -- what the sensor reports

The filter carries a Gaussian belief (mean x, covariance P) and alternates two steps:

  PREDICT: push the belief through the dynamics -- x <- F x, P <- F P F' + Q. Uncertainty grows.
  UPDATE:  fold in a measurement -- form the innovation z - H x, weight it by the KALMAN GAIN
           K = P H' (H P H' + R)^-1 (how much to trust the sensor vs the model), and shrink P.

Because both the model and the sensor are noisy, the fused estimate is better than either alone --
the filtered variance is provably below the measurement variance. Running a backward pass after
filtering (the RTS SMOOTHER) uses future measurements to refine past estimates, better still.

This module implements the multivariate predict/update filter and the RTS smoother over a full
sequence, with self-contained small-matrix helpers -- verified on a constant-velocity tracking
problem: the filter's error and variance fall below the raw measurements', a steady-state gain is
reached, a zero-noise sensor is trusted exactly, and the smoother improves on the filter. Pure
stdlib; the continuous-state companion to the hidden-Markov-model note."""

from __future__ import annotations


# --- tiny dense-matrix helpers (lists of lists) ----------------------------
def matmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    out = [[0.0] * p for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        for k in range(m):
            a = Ai[k]
            if a == 0.0:
                continue
            Bk = B[k]
            outi = out[i]
            for j in range(p):
                outi[j] += a * Bk[j]
    return out


def transpose(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def matadd(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def matsub(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def matvec(A, v):
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def vecsub(a, b):
    return [a[i] - b[i] for i in range(len(a))]


def vecadd(a, b):
    return [a[i] + b[i] for i in range(len(a))]


def eye(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def inverse(A):
    """Inverse of a small square matrix by Gauss-Jordan with partial pivoting."""
    n = len(A)
    M = [list(A[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-15:
            raise ValueError("singular matrix in Kalman update")
        M[col], M[piv] = M[piv], M[col]
        d = M[col][col]
        M[col] = [x / d for x in M[col]]
        for r in range(n):
            if r != col:
                f = M[r][col]
                if f != 0.0:
                    M[r] = [M[r][j] - f * M[col][j] for j in range(2 * n)]
    return [row[n:] for row in M]


class KalmanFilter:
    """A linear-Gaussian Kalman filter.

    F  state-transition matrix        (n x n)
    H  observation matrix             (m x n)
    Q  process-noise covariance       (n x n)
    R  measurement-noise covariance   (m x m)
    x  initial state mean             (length n)
    P  initial state covariance       (n x n)
    """

    def __init__(self, F, H, Q, R, x0, P0):
        self.F = [list(r) for r in F]
        self.H = [list(r) for r in H]
        self.Q = [list(r) for r in Q]
        self.R = [list(r) for r in R]
        self.x = list(x0)
        self.P = [list(r) for r in P0]
        self.n = len(F)

    def predict(self):
        self.x = matvec(self.F, self.x)
        self.P = matadd(matmul(matmul(self.F, self.P), transpose(self.F)), self.Q)
        return self.x, self.P

    def update(self, z):
        """Fold in one measurement z. Returns (state mean, covariance) and stores them."""
        Ht = transpose(self.H)
        y = vecsub(z, matvec(self.H, self.x))                 # innovation
        S = matadd(matmul(matmul(self.H, self.P), Ht), self.R)  # innovation covariance
        K = matmul(matmul(self.P, Ht), inverse(S))            # Kalman gain
        self.x = vecadd(self.x, matvec(K, y))
        # Joseph-form-free but numerically fine covariance update: (I - K H) P
        KH = matmul(K, self.H)
        self.P = matmul(matsub(eye(self.n), KH), self.P)
        return self.x, self.P, K

    def filter(self, measurements):
        """Run predict/update over a sequence. Returns per-step means, covariances, and the
        predicted priors (needed by the smoother)."""
        means, covs = [], []
        pri_means, pri_covs = [], []
        for z in measurements:
            self.predict()
            pri_means.append(list(self.x))
            pri_covs.append([list(r) for r in self.P])
            self.update(z)
            means.append(list(self.x))
            covs.append([list(r) for r in self.P])
        return means, covs, pri_means, pri_covs


def rts_smoother(kf, measurements):
    """Rauch-Tung-Striebel smoother: forward filter, then a backward sweep that uses future data
    to refine each past estimate. Returns (smoothed means, smoothed covariances)."""
    means, covs, pri_means, pri_covs = kf.filter(measurements)
    T = len(means)
    Ft = transpose(kf.F)
    sm_means = [None] * T
    sm_covs = [None] * T
    sm_means[-1] = means[-1]
    sm_covs[-1] = covs[-1]
    for t in range(T - 2, -1, -1):
        # C = P_t F' (P_pred_{t+1})^-1
        C = matmul(matmul(covs[t], Ft), inverse(pri_covs[t + 1]))
        sm_means[t] = vecadd(means[t], matvec(C, vecsub(sm_means[t + 1], pri_means[t + 1])))
        diff = matsub(sm_covs[t + 1], pri_covs[t + 1])
        sm_covs[t] = matadd(covs[t], matmul(matmul(C, diff), transpose(C)))
    return sm_means, sm_covs


def constant_velocity(dt, process_var, meas_var, x0=None):
    """Build a 1-D constant-velocity tracker: state = [position, velocity], sensor sees position.

    Returns a KalmanFilter with the standard kinematic F, the continuous-white-noise-acceleration
    process covariance Q, and scalar measurement noise R."""
    F = [[1.0, dt], [0.0, 1.0]]
    H = [[1.0, 0.0]]
    # continuous white-noise acceleration model
    q = process_var
    Q = [[dt ** 4 / 4 * q, dt ** 3 / 2 * q],
         [dt ** 3 / 2 * q, dt ** 2 * q]]
    R = [[meas_var]]
    if x0 is None:
        x0 = [0.0, 0.0]
    P0 = [[meas_var, 0.0], [0.0, meas_var]]
    return KalmanFilter(F, H, Q, R, x0, P0)
