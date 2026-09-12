"""The linear-quadratic regulator -- the optimal way to steer a linear system, from the Riccati equation.

You have a system you want to hold at a target: a drone hovering, a cart balancing an inverted pendulum,
a chemical reactor at setpoint, a satellite pointing. It drifts, and you can push on it. Push too gently
and it wanders; push too hard and you waste fuel and overshoot. The LINEAR-QUADRATIC REGULATOR answers,
exactly and optimally, how hard to push: for a system whose dynamics are linear and whose cost is
quadratic, it computes the control law that minimises total cost over all time. It is the foundation of
modern control theory and the inner loop of countless real controllers.

Set up the problem in discrete time. The state x_t evolves as x_{t+1} = A x_t + B u_t, where u_t is the
control you choose. You pay a quadratic cost each step: x^T Q x for being away from the target (Q weights
which state errors hurt) plus u^T R u for the effort spent (R weights how expensive control is). The LQR
finds the feedback law u_t = -K x_t that minimises the total discounted cost. The genius is that the
optimal K is CONSTANT -- a fixed gain matrix -- and it comes from the solution P of the discrete
algebraic RICCATI equation:

    P = Q + A^T P A - A^T P B (R + B^T P B)^-1 B^T P A,

after which K = (R + B^T P B)^-1 B^T P A. This module solves the Riccati equation by iterating that
fixed-point map to convergence, forms the gain, and simulates the closed-loop system so you can watch it
settle. It also gives the finite-horizon version (a backward Riccati recursion from a terminal cost) and
the closed-loop matrix A - B K whose eigenvalues reveal stability.

All linear algebra -- small dense matrix multiply, transpose, and inverse by Gauss-Jordan -- is written
out by hand, pure standard library.

Validation. (1) The infinite-horizon P must SATISFY the algebraic Riccati equation to numerical
precision (plug it back in; the residual is ~0). (2) The closed-loop system is STABLE: every eigenvalue
of A - B K lies strictly inside the unit circle, and a simulation from any initial state decays to the
origin. (3) OPTIMALITY on a scalar system, where the Riccati equation is a solvable quadratic: the code's
P and K match the closed-form answer exactly. (4) The realised closed-loop cost is no larger than that of
many randomly perturbed stabilising gains -- the LQR gain is the minimiser. (5) Heavier control penalty R
yields a gentler gain (less aggressive control), and heavier state penalty Q yields a stronger one, the
expected monotone response. (6) The finite-horizon recursion converges to the infinite-horizon gain as
the horizon grows. Pure standard library."""

import math


# ---------------------------------------------------------------------------
# dense linear algebra
# ---------------------------------------------------------------------------

def _mm(A, B):
    n, m, p = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]


def _T(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def _add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _sub(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _matvec(A, v):
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def _inverse(A):
    n = len(A)
    M = [list(A[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-15:
            raise ValueError("singular matrix in LQR solve")
        M[col], M[piv] = M[piv], M[col]
        p = M[col][col]
        M[col] = [x / p for x in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0:
                f = M[r][col]
                M[r] = [M[r][k] - f * M[col][k] for k in range(2 * n)]
    return [row[n:] for row in M]


def _frob_diff(A, B):
    return max(abs(A[i][j] - B[i][j]) for i in range(len(A)) for j in range(len(A[0])))


# ---------------------------------------------------------------------------
# discrete algebraic Riccati equation and the LQR gain
# ---------------------------------------------------------------------------

def solve_dare(A, B, Q, R, max_iters=10000, tol=1e-12):
    """Solve the discrete algebraic Riccati equation by fixed-point iteration. Returns P."""
    n = len(A)
    P = [row[:] for row in Q]
    At = _T(A)
    Bt = _T(B)
    for _ in range(max_iters):
        # S = R + B^T P B
        BtP = _mm(Bt, P)
        S = _add(R, _mm(BtP, B))
        Sinv = _inverse(S)
        # AtPA - AtPB Sinv BtPA + Q
        AtP = _mm(At, P)
        AtPA = _mm(AtP, A)
        AtPB = _mm(AtP, B)
        BtPA = _mm(BtP, A)
        correction = _mm(_mm(AtPB, Sinv), BtPA)
        P_next = _add(_sub(AtPA, correction), Q)
        if _frob_diff(P_next, P) < tol:
            return P_next
        P = P_next
    return P


def lqr_gain(A, B, Q, R):
    """Optimal infinite-horizon LQR feedback gain K (so u = -K x). Returns (K, P)."""
    P = solve_dare(A, B, Q, R)
    Bt = _T(B)
    S = _add(R, _mm(_mm(Bt, P), B))
    K = _mm(_mm(_inverse(S), Bt), _mm(P, A))
    return K, P


def finite_horizon_gains(A, B, Q, R, horizon, Qf=None):
    """Time-varying LQR gains for a finite horizon via backward Riccati recursion.

    Returns a list of gains K_0..K_{horizon-1} (K_0 applied first). ``Qf`` is the terminal cost (Q if
    omitted).
    """
    At, Bt = _T(A), _T(B)
    P = [row[:] for row in (Qf if Qf is not None else Q)]
    gains = []
    for _ in range(horizon):
        BtP = _mm(Bt, P)
        S = _add(R, _mm(BtP, B))
        Sinv = _inverse(S)
        K = _mm(_mm(Sinv, Bt), _mm(P, A))
        gains.append(K)
        # update P backwards
        AtP = _mm(At, P)
        AtPA = _mm(AtP, A)
        AtPB = _mm(AtP, B)
        BtPA = _mm(BtP, A)
        P = _add(_sub(AtPA, _mm(_mm(AtPB, Sinv), BtPA)), Q)
    gains.reverse()   # so gains[0] is applied at t=0
    return gains


# ---------------------------------------------------------------------------
# simulation and analysis
# ---------------------------------------------------------------------------

def closed_loop_matrix(A, B, K):
    """The closed-loop dynamics matrix A - B K."""
    return _sub(A, _mm(B, K))


def simulate(A, B, K, x0, steps):
    """Simulate x_{t+1} = (A - B K) x_t from x0. Returns the list of states x_0..x_steps."""
    Acl = closed_loop_matrix(A, B, K)
    xs = [list(x0)]
    x = list(x0)
    for _ in range(steps):
        x = _matvec(Acl, x)
        xs.append(x)
    return xs


def lqr_cost(A, B, K, Q, R, x0, steps):
    """Total quadratic cost sum(x^T Q x + u^T R u) of running gain K from x0 for ``steps`` steps."""
    x = list(x0)
    total = 0.0
    for _ in range(steps):
        u = [-sum(K[i][j] * x[j] for j in range(len(x))) for i in range(len(K))]
        total += sum(x[i] * sum(Q[i][j] * x[j] for j in range(len(x))) for i in range(len(x)))
        total += sum(u[i] * sum(R[i][j] * u[j] for j in range(len(u))) for i in range(len(u)))
        # advance
        Ax = _matvec(A, x)
        Bu = _matvec(B, u)
        x = [Ax[i] + Bu[i] for i in range(len(x))]
    return total


def spectral_radius_2x2(M):
    """Magnitude of the largest eigenvalue of a 2x2 matrix (for stability checks)."""
    a, b = M[0][0], M[0][1]
    c, d = M[1][0], M[1][1]
    tr = a + d
    det = a * d - b * c
    disc = tr * tr - 4 * det
    if disc >= 0:
        r = math.sqrt(disc)
        return max(abs((tr + r) / 2), abs((tr - r) / 2))
    # complex conjugate pair: magnitude sqrt(det)
    return math.sqrt(det)
