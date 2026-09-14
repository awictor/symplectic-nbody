"""Tests for Dirichlet convolution: the classical identities, commutativity/associativity, inversion."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dirichlet import (  # noqa: E402
    dirichlet_convolution, dirichlet_inverse,
    epsilon, one, identity, mobius, totient, divisor_count, divisor_sum,
    mobius_inversion, divisor_sum_transform,
    brute_divisor_count, brute_divisor_sum,
)


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


def _eq(a, b, N):
    return all(a[n] == b[n] for n in range(1, N + 1))


def main():
    N = 500

    eps = epsilon(N)
    ones = one(N)
    idf = identity(N)
    mu = mobius(N)
    phi = totient(N)

    # ---- 1. mu * 1 = epsilon (defining property of Mobius) ------------------------------
    check("mu * 1 = epsilon", _eq(dirichlet_convolution(mu, ones, N), eps, N))

    # ---- 2. phi * 1 = id (sum of totient over divisors is n) ----------------------------
    check("phi * 1 = id", _eq(dirichlet_convolution(phi, ones, N), idf, N))

    # ---- 3. 1 * 1 = d (divisor count) ---------------------------------------------------
    d = dirichlet_convolution(ones, ones, N)
    check("1 * 1 = divisor count", all(d[n] == brute_divisor_count(n) for n in range(1, N + 1)))
    check("divisor_count helper == brute", all(divisor_count(N)[n] == brute_divisor_count(n)
                                               for n in range(1, N + 1)))

    # ---- 4. id * 1 = sigma (divisor sum) ------------------------------------------------
    sigma = dirichlet_convolution(idf, ones, N)
    check("id * 1 = sigma", all(sigma[n] == brute_divisor_sum(n) for n in range(1, N + 1)))
    check("divisor_sum helper == brute", all(divisor_sum(N)[n] == brute_divisor_sum(n)
                                             for n in range(1, N + 1)))

    # ---- 5. mu * id = phi (Mobius inversion form) ---------------------------------------
    check("mu * id = phi", _eq(dirichlet_convolution(mu, idf, N), phi, N))

    # ---- 6. epsilon is the convolution identity -----------------------------------------
    check("f * epsilon = f (for phi)", _eq(dirichlet_convolution(phi, eps, N), phi, N))
    check("f * epsilon = f (for mu)", _eq(dirichlet_convolution(mu, eps, N), mu, N))

    # ---- 7. convolution is commutative --------------------------------------------------
    check("f * g = g * f", _eq(dirichlet_convolution(phi, mu, N),
                               dirichlet_convolution(mu, phi, N), N))

    # ---- 8. convolution is associative --------------------------------------------------
    lhs = dirichlet_convolution(dirichlet_convolution(mu, ones, N), phi, N)
    rhs = dirichlet_convolution(mu, dirichlet_convolution(ones, phi, N), N)
    check("(f*g)*h = f*(g*h)", _eq(lhs, rhs, N))

    # ---- 9. Dirichlet inverse of 1 is mu ------------------------------------------------
    inv1 = dirichlet_inverse(ones, N)
    check("Dirichlet inverse of 1 is mu", all(inv1[n] == mu[n] for n in range(1, N + 1)))

    # ---- 10. f * f^{-1} = epsilon -------------------------------------------------------
    inv_phi = dirichlet_inverse(phi, N)
    # convolve phi (as Fraction-compatible ints) with its inverse
    conv = [0] * (N + 1)
    for dd in range(1, N + 1):
        for m in range(dd, N + 1, dd):
            conv[m] += phi[dd] * inv_phi[m // dd]
    check("phi * phi^{-1} = epsilon", conv[1] == 1 and all(conv[n] == 0 for n in range(2, N + 1)))

    # ---- 11. Mobius inversion round-trips: (f * 1) * mu = f ------------------------------
    # take a random-ish f and confirm recovery
    f = [0] + [((n * 7 + 3) % 11) for n in range(1, N + 1)]
    F = divisor_sum_transform(f, N)
    recovered = mobius_inversion(F, N)
    check("Mobius inversion recovers f", _eq(recovered, f, N))

    # ---- 12. sigma_0 = divisor count, sigma_2 correct -----------------------------------
    check("sigma_0 = d", all(divisor_sum(N, 0)[n] == brute_divisor_count(n) for n in range(1, N + 1)))
    check("sigma_2 == brute", all(divisor_sum(100, 2)[n] == brute_divisor_sum(n, 2)
                                  for n in range(1, 101)))

    # ---- 13. small explicit values ------------------------------------------------------
    # d(12) = 6 (1,2,3,4,6,12), sigma(12) = 28, phi(12) = 4, mu(12) = 0
    check("d(12) = 6", divisor_count(12)[12] == 6)
    check("sigma(12) = 28", divisor_sum(12)[12] == 28)
    check("phi(12) = 4", phi[12] == 4)
    check("mu(12) = 0", mu[12] == 0)
    check("mu(30) = -1 (three prime factors)", mu[30] == -1)
    check("mu(1) = 1", mu[1] == 1)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
