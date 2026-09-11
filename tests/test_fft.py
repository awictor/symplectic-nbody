"""Tests for fft.py -- the radix-2 Fast Fourier Transform.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. The FFT is cross-checked against
a naive DFT, its inverse round-trip, the convolution theorem, and Parseval's identity.
"""

import cmath
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import fft as F  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def close(a, b, tol=1e-9):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


# --- FFT matches the naive DFT ---------------------------------------------
x = [1, 2, 3, 4, 5, 6, 7, 8]
check("FFT matches DFT on a real signal", close(F.fft(x), F.dft(x)))
cx = [complex(i, n - i) for i, n in enumerate(range(8, 0, -1))]
check("FFT matches DFT on a complex signal", close(F.fft(cx), F.dft(cx)))
check("FFT of a single element is itself", F.fft([5]) == [complex(5)])
check("FFT of empty is empty", F.fft([]) == [])
try:
    F.fft([1, 2, 3])       # not a power of two
    check("FFT rejects non-power-of-two length", False)
except ValueError:
    check("FFT rejects non-power-of-two length", True)

# --- inverse round-trip -----------------------------------------------------
check("ifft(fft(x)) recovers x", close([complex(v) for v in x], F.ifft(F.fft(x))))
check("round-trip on random-ish data",
      close([complex(v) for v in [3, -1, 4, 1, 5, -9, 2, 6]],
            F.ifft(F.fft([3, -1, 4, 1, 5, -9, 2, 6]))))
check("ifft of empty is empty", F.ifft([]) == [])

# --- known transforms -------------------------------------------------------
# DC (constant) signal: all energy in bin 0
dc = F.fft([1, 1, 1, 1])
check("DC signal puts all energy in bin 0", abs(dc[0] - 4) < 1e-9 and all(abs(dc[k]) < 1e-9 for k in (1, 2, 3)))
# a pure cosine at integer frequency f -> spikes at bins f and n-f
n, f = 64, 8
cos_sig = [math.cos(2 * math.pi * f * j / n) for j in range(n)]
mag = F.magnitude_spectrum(cos_sig)
check("a pure cosine peaks at its frequency bin", mag[f] > 0.9 * max(mag))
check("cosine spectrum is symmetric (real signal)", abs(mag[f] - mag[n - f]) < 1e-6)
check("cosine amplitude n/2 at the peak", abs(mag[f] - n / 2) < 1e-6)
# a sine at frequency f: same peak location, imaginary
sin_sig = [math.sin(2 * math.pi * f * j / n) for j in range(n)]
smag = F.magnitude_spectrum(sin_sig)
check("a pure sine also peaks at its frequency", smag[f] > 0.9 * max(smag))

# --- Parseval's theorem -----------------------------------------------------
X = F.fft(x)
time_energy = sum(abs(complex(v)) ** 2 for v in x)
freq_energy = sum(abs(v) ** 2 for v in X) / len(x)
check("Parseval: time energy equals freq energy / n", abs(time_energy - freq_energy) < 1e-9)

# --- linearity --------------------------------------------------------------
a = [1, 0, 0, 0, 2, 0, 0, 0]
b = [0, 1, 0, 1, 0, 1, 0, 1]
fa, fb = F.fft(a), F.fft(b)
fsum = F.fft([ai + bi for ai, bi in zip(a, b)])
check("FFT is linear", close(fsum, [x + y for x, y in zip(fa, fb)]))

# --- convolution theorem ----------------------------------------------------
def direct_convolve(u, v):
    m = len(u) + len(v) - 1
    out = [0.0] * m
    for i in range(len(u)):
        for j in range(len(v)):
            out[i + j] += u[i] * v[j]
    return out


ca, cb = [1, 2, 3], [4, 5, 6]
check("FFT convolution matches direct", close(F.convolve(ca, cb), direct_convolve(ca, cb), 1e-9))
check("convolution result length is len(a)+len(b)-1", len(F.convolve([1, 2, 3, 4], [1, 1])) == 5)
check("convolving with [1] is the identity", close(F.convolve([3, 1, 4], [1]), [3, 1, 4], 1e-9))
check("empty convolution is empty", F.convolve([], [1, 2]) == [])
# polynomial multiplication: (1 + 2x)(3 + 4x) = 3 + 10x + 8x^2
check("convolution multiplies polynomials", close(F.convolve([1, 2], [3, 4]), [3, 10, 8], 1e-9))
# larger random convolution
import random
random.seed(1)
u = [random.uniform(-5, 5) for _ in range(20)]
v = [random.uniform(-5, 5) for _ in range(13)]
check("FFT convolution matches direct on larger random inputs",
      close(F.convolve(u, v), direct_convolve(u, v), 1e-6))

# --- spectra helpers --------------------------------------------------------
check("power spectrum is magnitude squared",
      close(F.power_spectrum(x), [m * m for m in F.magnitude_spectrum(x)], 1e-9))
check("frequency bins scale with sample rate",
      F.frequencies(8, sample_rate=16.0) == [k * 2.0 for k in range(8)])
check("bin 0 is DC (0 Hz)", F.frequencies(8, 44100)[0] == 0.0)

# --- a two-tone signal shows two peaks -------------------------------------
n = 128
two = [math.cos(2 * math.pi * 5 * j / n) + 0.5 * math.cos(2 * math.pi * 20 * j / n) for j in range(n)]
tmag = F.magnitude_spectrum(two)
check("two-tone signal peaks at both frequencies",
      tmag[5] > 0.4 * max(tmag) and tmag[20] > 0.15 * max(tmag))
check("the stronger tone has the larger peak", tmag[5] > tmag[20])


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall fft tests passed")
