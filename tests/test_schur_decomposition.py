"""Tests for real Schur decomposition: A=QTQ^T, quasi-triangular T, eigenvalues, symmetric->diagonal."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import schur_decomposition as SD  # noqa: E402
from qr_algorithm import eigenvalues as qeig  # noqa: E402


PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def _lcg(seed):
    st = seed & 0xFFFFFFFF

    def rnd():
        nonlocal st
        st = (1664525 * st + 1013904223) & 0xFFFFFFFF
        return (st >> 8) / (1 << 24)

    return rnd


def _maxerr(A, B):
    return max(abs(A[i][j] - B[i][j]) for i in range(len(A)) for j in range(len(A[0])))


def _is_orth(Q, tol=1e-8):
    n = len(Q)
    return all(abs(sum(Q[k][i] * Q[k][j] for k in range(n)) - (1.0 if i == j else 0.0)) < tol
               for i in range(n) for j in range(n))


def main():
    rnd = _lcg(7)
    n = 5
    A = [[rnd() * 2 - 1 for _ in range(n)] for _ in range(n)]
    Q, T = SD.schur(A)

    # ---- 1. Q orthogonal, T quasi-upper-triangular, reconstruction ----------------------
    check("Q is orthogonal", _is_orth(Q))
    check("T is quasi-upper-triangular", SD.is_quasi_upper_triangular(T))
    check("Q T Q^T reconstructs A", _maxerr(SD.reconstruct(Q, T), A) < 1e-9,
          f"{_maxerr(SD.reconstruct(Q, T), A):.2e}")

    # ---- 2. eigenvalues from Schur match an independent solver (incl. complex) ----------
    es = sorted(SD.eigenvalues_from_schur(T), key=lambda z: (round(z.real, 5), round(z.imag, 5)))
    eq = sorted(qeig(A), key=lambda z: (round(z.real, 5), round(z.imag, 5)))
    check("Schur eigenvalues match QR-algorithm",
          all(abs(es[i] - eq[i]) < 1e-5 for i in range(n)), f"{es}\n{eq}")

    # ---- 3. eigenvalues on the diagonal match the trace and determinant -----------------
    total = sum(z.real for z in es)
    check("sum of eigenvalues == trace", abs(total - sum(A[i][i] for i in range(n))) < 1e-6)
    prod = 1.0 + 0j
    for z in es:
        prod *= z
    from lu import determinant
    check("product of eigenvalues == determinant", abs(prod.real - determinant(A)) < 1e-5,
          f"{prod.real:.4f} vs {determinant(A):.4f}")

    # ---- 4. symmetric matrix -> genuinely diagonal T (all real eigenvalues) -------------
    S = [[A[i][j] + A[j][i] for j in range(n)] for i in range(n)]
    Qs, Ts = SD.schur(S)
    off = max(abs(Ts[i][j]) for i in range(n) for j in range(n) if i != j)
    check("symmetric input -> diagonal T", off < 1e-6, f"max off-diag {off:.2e}")
    check("symmetric reconstruction", _maxerr(SD.reconstruct(Qs, Ts), S) < 1e-9)
    # diagonal entries are the (real) eigenvalues
    diag = sorted(Ts[i][i] for i in range(n))
    ref = sorted(z.real for z in qeig(S))
    check("symmetric diagonal == eigenvalues",
          all(abs(diag[i] - ref[i]) < 1e-6 for i in range(n)), f"{diag}\n{ref}")

    # ---- 5. a matrix with a genuine complex pair keeps a 2x2 block ----------------------
    # rotation-scaling block has complex eigenvalues
    C = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]]
    Qc, Tc = SD.schur(C)
    ec = SD.eigenvalues_from_schur(Tc)
    has_complex = any(abs(z.imag) > 1e-6 for z in ec)
    check("complex-eigenvalue matrix keeps a 2x2 block", has_complex, f"{ec}")
    check("complex case reconstructs", _maxerr(SD.reconstruct(Qc, Tc), C) < 1e-9)
    # the eigenvalues should be +/- i and 2
    check("complex eigenvalues are +/- i and 2",
          any(abs(z - complex(0, 1)) < 1e-6 for z in ec) and
          any(abs(z - complex(0, -1)) < 1e-6 for z in ec) and
          any(abs(z - 2) < 1e-6 for z in ec), f"{ec}")

    # ---- 6. diagonal matrix is its own Schur form ---------------------------------------
    D = [[float(i + 1) if i == j else 0.0 for j in range(4)] for i in range(4)]
    Qd, Td = SD.schur(D)
    check("diagonal matrix -> diagonal T", _maxerr(Td, D) < 1e-9)

    # ---- 7. upper-triangular matrix stays triangular, diagonal = eigenvalues ------------
    U = [[2.0, 3.0, 1.0], [0.0, 5.0, -2.0], [0.0, 0.0, 7.0]]
    Qu, Tu = SD.schur(U)
    eu = sorted(z.real for z in SD.eigenvalues_from_schur(Tu))
    check("triangular eigenvalues are the diagonal 2,5,7", eu == sorted([2.0, 5.0, 7.0]) or
          all(abs(eu[i] - [2.0, 5.0, 7.0][i]) < 1e-6 for i in range(3)), f"{eu}")

    # ---- 8. larger random matrix ---------------------------------------------------------
    rnd2 = _lcg(123)
    m = 8
    B = [[rnd2() * 2 - 1 for _ in range(m)] for _ in range(m)]
    Qb, Tb = SD.schur(B)
    check("8x8 reconstruction", _maxerr(SD.reconstruct(Qb, Tb), B) < 1e-7,
          f"{_maxerr(SD.reconstruct(Qb, Tb), B):.2e}")
    check("8x8 quasi-triangular", SD.is_quasi_upper_triangular(Tb))
    eb = sorted(SD.eigenvalues_from_schur(Tb), key=lambda z: (round(z.real, 4), round(z.imag, 4)))
    ebq = sorted(qeig(B), key=lambda z: (round(z.real, 4), round(z.imag, 4)))
    check("8x8 eigenvalues match", all(abs(eb[i] - ebq[i]) < 1e-4 for i in range(m)))

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
