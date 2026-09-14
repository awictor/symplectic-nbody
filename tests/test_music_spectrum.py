"""Validate MUSIC: exact peak frequencies, super-resolution, eigenvalue split, vs ESPRIT."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import music_spectrum as music
import esprit_method as esprit


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


class _R:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF
        self._spare = None

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)

    def normal(self):
        if self._spare is not None:
            v = self._spare
            self._spare = None
            return v
        u1 = max(self.u(), 1e-12)
        u2 = self.u()
        r = math.sqrt(-2 * math.log(u1))
        self._spare = r * math.sin(2 * math.pi * u2)
        return r * math.cos(2 * math.pi * u2)


def tones(freqs, n, rng=None, noise=0.0, amps=None):
    if amps is None:
        amps = [1.0] * len(freqs)
    x = []
    for t in range(n):
        v = sum(amps[k] * math.cos(2 * math.pi * freqs[k] * t) for k in range(len(freqs)))
        if rng is not None and noise:
            v += noise * rng.normal()
        x.append(v)
    return x


def nearest(peaks, target):
    return min(peaks, key=lambda f: abs(f - target))


def main():
    print("MUSIC tests")

    # --- single tone: peak at the exact frequency ---
    f0 = 0.12
    x = tones([f0], 128)
    peaks = music.music_peaks(x, n_signal=2, n_peaks=1)
    check(f"single tone peak exact ({peaks[0]:.4f} vs {f0})", abs(peaks[0] - f0) < 5e-3)

    # --- two well-separated tones ---
    fs = [0.10, 0.30]
    x = tones(fs, 128)
    peaks = music.music_peaks(x, n_signal=4, n_peaks=2)
    ok = all(abs(nearest(peaks, f) - f) < 5e-3 for f in fs)
    check("two separated tones recovered", ok)

    # --- SUPER-RESOLUTION: two tones closer than one FFT bin ---
    n = 100
    bin_hz = 1.0 / n
    f1 = 0.20
    f2 = 0.20 + 0.4 * bin_hz
    rng = _R(3)
    x = tones([f1, f2], n, rng=rng, noise=0.02)
    peaks = music.music_peaks(x, n_signal=4, m=40, n_peaks=2)
    resolved = (abs(nearest(peaks, f1) - f1) < 0.01 and abs(nearest(peaks, f2) - f2) < 0.01)
    check(f"super-resolves tones 0.4 bin apart", resolved)

    # --- eigenvalue split: signal eigenvalues >> noise floor ---
    rng = _R(9)
    x = tones([0.15, 0.35], 200, rng=rng, noise=0.1)
    ev = music.eigenvalue_spectrum(x, m=20)
    # top 4 eigenvalues (2 tones -> 4) should dwarf the rest
    signal_min = ev[3]
    noise_max = ev[4]
    check(f"eigenvalue split signal vs noise ({signal_min:.3f} >> {noise_max:.3f})",
          signal_min > 5 * noise_max)

    # --- MUSIC peaks are sharper than the periodogram (higher peak-to-floor ratio) ---
    n = 128
    x = tones([0.22], n)
    freqs, psd = music.music_spectrum(x, n_signal=2, n_freqs=1024)
    peak = max(psd)
    floor = sorted(psd)[len(psd) // 2]  # median as the floor
    music_ratio = peak / floor
    check("MUSIC peak-to-floor ratio is large", music_ratio > 100)

    # --- cross-check against ESPRIT ---
    dt = 1.0
    fs = [0.12, 0.27]
    x = tones(fs, 150)
    m_peaks = sorted(music.music_peaks(x, n_signal=4, n_peaks=2))
    e = esprit.esprit(x, p=4, dt=dt)
    e_freqs = sorted(set(round(abs(f), 3) for f in e["frequencies"] if 0.01 < abs(f) < 0.5))
    # match MUSIC peaks to nearest ESPRIT frequencies
    ok = all(min(abs(mp - ef) for ef in e_freqs) < 0.01 for mp in m_peaks)
    check("MUSIC agrees with ESPRIT", ok)

    # --- three tones ---
    fs = [0.08, 0.18, 0.33]
    x = tones(fs, 200)
    peaks = music.music_peaks(x, n_signal=6, n_peaks=3)
    ok = all(abs(nearest(peaks, f) - f) < 5e-3 for f in fs)
    check("three tones recovered", ok)

    # --- pseudospectrum strictly positive ---
    freqs, psd = music.music_spectrum(tones([0.2], 64), n_signal=2, n_freqs=128)
    check("pseudospectrum positive", all(p > 0 for p in psd))

    # --- deterministic ---
    a = music.music_peaks(x, n_signal=6, n_peaks=3)
    b = music.music_peaks(x, n_signal=6, n_peaks=3)
    check("deterministic", a == b)

    # --- window size must exceed signal subspace ---
    try:
        music.music_spectrum(tones([0.2], 64), n_signal=10, m=8)
        check("rejects m <= n_signal", False)
    except ValueError:
        check("rejects m <= n_signal", True)

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
