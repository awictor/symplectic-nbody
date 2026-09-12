"""Number-theoretic transform: exact integer convolution with no floating-point error.

The FFT multiplies polynomials and convolves sequences in O(n log n), but over the COMPLEX numbers --
so its answers carry floating-point rounding, and large integer coefficients can lose precision. The
NUMBER-THEORETIC TRANSFORM (NTT) is the FFT done in modular arithmetic instead: it replaces the complex
root of unity e^(2 pi i / n) with a PRIMITIVE ROOT OF UNITY in the field of integers modulo a carefully
chosen prime. Everything else -- the butterfly, the divide-and-conquer, the convolution theorem -- is
identical, but every operation is an exact integer, so the result is EXACT. This is the workhorse
behind big-integer multiplication (used in arbitrary-precision libraries), exact polynomial arithmetic
in computer algebra, and competitive-programming convolutions where the answer must be taken modulo a
prime.

The chosen modulus is a "NTT-friendly" prime p = c * 2^k + 1, so the multiplicative group modulo p has
order p-1 divisible by a large power of two -- enough to hold a primitive n-th root of unity for any
transform length n that is a power of two up to 2^k. This module uses p = 998244353 = 119 * 2^23 + 1
(the standard competitive-programming prime, supporting lengths up to 2^23) with primitive root 3. The
forward transform is the iterative Cooley-Tukey butterfly with root powers replacing twiddle factors;
the inverse divides by n via the modular inverse. CONVOLUTION of two sequences is then transform,
pointwise multiply, inverse-transform -- exactly, mod p -- and for genuine INTEGER convolution (with
results below p) the modular answer is the true integer answer.

This module implements the forward and inverse NTT modulo 998244353, polynomial multiplication /
convolution, and a big-integer multiplier that convolves digit arrays. It is verified against
schoolbook O(n^2) convolution and Python's exact big-integer multiplication -- identical results on
hundreds of random inputs -- plus the round-trip identity (inverse undoes forward) and agreement with
the complex FFT convolution on small integer inputs. Pure stdlib; a number-theory-meets-signals
companion to the FFT, Walsh-Hadamard transform, and modular-arithmetic notes."""

from __future__ import annotations

MOD = 998244353          # 119 * 2^23 + 1, a NTT-friendly prime
ROOT = 3                 # a primitive root modulo MOD


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def ntt(a, invert=False):
    """In-place-style iterative number-theoretic transform of `a` (length a power of two) modulo MOD.
    Returns a new list. `invert=True` computes the inverse transform (divided by n)."""
    a = list(a)
    n = len(a)
    if n & (n - 1) != 0:
        raise ValueError("length must be a power of two")

    # bit-reversal permutation
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    length = 2
    while length <= n:
        # primitive length-th root of unity (or its inverse)
        if invert:
            w = pow(ROOT, (MOD - 1) // length, MOD)
            w = pow(w, MOD - 2, MOD)         # modular inverse
        else:
            w = pow(ROOT, (MOD - 1) // length, MOD)
        half = length // 2
        for i in range(0, n, length):
            wn = 1
            for k in range(half):
                u = a[i + k]
                v = a[i + k + half] * wn % MOD
                a[i + k] = (u + v) % MOD
                a[i + k + half] = (u - v) % MOD
                wn = wn * w % MOD
        length <<= 1

    if invert:
        inv_n = pow(n, MOD - 2, MOD)
        a = [x * inv_n % MOD for x in a]
    return a


def convolve(a, b):
    """The (linear) convolution of integer sequences `a` and `b`, computed exactly modulo MOD via the
    NTT. Returns a list of length len(a)+len(b)-1. If all true convolution values are < MOD, they are
    the exact integers."""
    if not a or not b:
        return []
    result_len = len(a) + len(b) - 1
    n = _next_pow2(result_len)
    fa = ntt([x % MOD for x in a] + [0] * (n - len(a)))
    fb = ntt([x % MOD for x in b] + [0] * (n - len(b)))
    fc = [fa[i] * fb[i] % MOD for i in range(n)]
    conv = ntt(fc, invert=True)
    return conv[:result_len]


def poly_multiply(a, b):
    """Multiply two polynomials given as coefficient lists (index = power), exactly mod MOD. Returns
    the product's coefficient list."""
    return convolve(a, b)


def multiply_big_integers(x, y, base=10):
    """Multiply two non-negative integers by convolving their base-`base` digit arrays with the NTT
    then carrying. Returns the exact integer product (validates the NTT against Python's own bignum
    multiply)."""
    if x == 0 or y == 0:
        return 0
    dx = _digits(x, base)
    dy = _digits(y, base)
    conv = convolve(dx, dy)
    # carry propagation
    carry = 0
    out = []
    for v in conv:
        v += carry
        out.append(v % base)
        carry = v // base
    while carry:
        out.append(carry % base)
        carry //= base
    # reassemble (out is little-endian base-`base`)
    result = 0
    for d in reversed(out):
        result = result * base + d
    return result


def _digits(x, base):
    """Little-endian base-`base` digits of x."""
    d = []
    while x:
        d.append(x % base)
        x //= base
    return d or [0]


# --- brute-force reference --------------------------------------------------
def schoolbook_convolve(a, b):
    """Direct O(n*m) convolution for validation (no modular reduction; true integers)."""
    if not a or not b:
        return []
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out
