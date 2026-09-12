"""Tests for poisson: harmonic solutions, mean-value property, method convergence speed."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from poisson import solve, residual, mean_value_error

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


# --- Laplace with linear boundary reproduces the exact linear solution -----
# u = i is harmonic (laplacian = 0); linear boundary -> exact solution u(i,j) = i
u, sweeps = solve(25, 25, boundary=lambda i, j: float(i), method="sor", tol=1e-10)
max_err = max(abs(u[i][j] - i) for i in range(25) for j in range(25))
check(f"Laplace with linear boundary matches u=i exactly (err {max_err:.2e})", max_err < 1e-6)

# u = i + 2j is also harmonic
u2, _ = solve(20, 20, boundary=lambda i, j: i + 2.0 * j, method="sor", tol=1e-10)
err2 = max(abs(u2[i][j] - (i + 2 * j)) for i in range(20) for j in range(20))
check(f"Laplace with bilinear-boundary u=i+2j matches exactly (err {err2:.2e})", err2 < 1e-6)

# --- the discrete Laplace solution satisfies the mean-value property -------
check("interior cells equal the average of their neighbours (harmonic)",
      mean_value_error(u) < 1e-6)

# --- no interior extrema (maximum principle) -------------------------------
# a harmonic function's interior max/min occur on the boundary
u3, _ = solve(20, 20, boundary=lambda i, j: math.sin(i * 0.3) + math.cos(j * 0.4), method="sor", tol=1e-10)
boundary_vals = [u3[i][j] for i in range(20) for j in range(20)
                 if i in (0, 19) or j in (0, 19)]
interior_vals = [u3[i][j] for i in range(1, 19) for j in range(1, 19)]
check("interior max does not exceed the boundary max (maximum principle)",
      max(interior_vals) <= max(boundary_vals) + 1e-6)
check("interior min is not below the boundary min",
      min(interior_vals) >= min(boundary_vals) - 1e-6)

# --- a known separable analytic harmonic solution --------------------------
# u(x,y) = sinh(k x) sin(k y) is harmonic. On an N x N grid with unit spacing, set k so k*(N-1) is
# modest, impose the exact solution on the boundary, and check the interior matches.
N = 22
k = math.pi / (N - 1)
def exact(i, j):
    return math.sinh(k * i) * math.sin(k * j)

ua, _ = solve(N, N, boundary=exact, method="sor", tol=1e-11, max_iter=200000)
# discretization error is O(h^2); compare relative to the solution's scale
scale = max(abs(exact(i, j)) for i in range(N) for j in range(N))
rel_err = max(abs(ua[i][j] - exact(i, j)) for i in range(1, N - 1) for j in range(1, N - 1)) / scale
check(f"separable harmonic sinh(kx)sin(ky) matched to grid accuracy (rel err {rel_err:.2e})",
      rel_err < 0.02)

# --- all three methods converge to the SAME field --------------------------
uj, sj = solve(20, 20, boundary=lambda i, j: float(i), method="jacobi", tol=1e-9)
ug, sg = solve(20, 20, boundary=lambda i, j: float(i), method="gauss_seidel", tol=1e-9)
us, ss = solve(20, 20, boundary=lambda i, j: float(i), method="sor", tol=1e-9)
same = all(abs(uj[i][j] - us[i][j]) < 1e-5 and abs(ug[i][j] - us[i][j]) < 1e-5
           for i in range(20) for j in range(20))
check("Jacobi, Gauss-Seidel, and SOR converge to the same field", same)

# --- convergence speed: SOR < Gauss-Seidel < Jacobi ------------------------
check(f"Gauss-Seidel converges faster than Jacobi ({sg} < {sj})", sg < sj)
check(f"SOR converges faster than Gauss-Seidel ({ss} < {sg})", ss < sg)
check(f"SOR is much faster than Jacobi ({ss} vs {sj})", ss < sj / 5)

# --- Poisson with a source: residual is near zero --------------------------
# solve laplacian(u) = 1 with zero boundary
up, _ = solve(25, 25, boundary=lambda i, j: 0.0, source=lambda i, j: 1.0, method="sor", tol=1e-9)
check("Poisson solution has near-zero residual", residual(up, source=lambda i, j: 1.0) < 1e-5)
# the solution should be negative inside (source pushes it down, like -potential of a charge)
check("uniform positive source gives a negative interior (bowl shape)",
      min(up[i][j] for i in range(1, 24) for j in range(1, 24)) < 0)

# --- a point charge: symmetric potential -----------------------------------
mid = 12
def point_source(i, j):
    return 1.0 if (i, j) == (mid, mid) else 0.0

uc, _ = solve(25, 25, boundary=lambda i, j: 0.0, source=point_source, method="sor", tol=1e-9)
# symmetry: u should be symmetric about the center
sym_ok = all(abs(uc[mid + d][mid] - uc[mid - d][mid]) < 1e-4 for d in range(1, 10))
check("point-source potential is symmetric about the center", sym_ok)
check("point-source potential is monotone decreasing away from the source",
      uc[mid][mid] < uc[mid + 1][mid] and uc[mid + 1][mid] < uc[mid + 2][mid])

# --- constant boundary -> constant field -----------------------------------
uconst, _ = solve(15, 15, boundary=lambda i, j: 5.0, method="sor", tol=1e-10)
check("constant boundary gives a constant field",
      all(abs(uconst[i][j] - 5.0) < 1e-6 for i in range(15) for j in range(15)))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all poisson tests passed")
