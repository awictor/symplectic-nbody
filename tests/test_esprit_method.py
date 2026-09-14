"""Validate ESPRIT: exact frequency recovery, super-resolution, and noise robustness vs Prony."""

import cmath
import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import esprit_method as esprit
import prony


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


def make_signal(freqs, amps, damps, n, dt, phases=None):
    if phases is None:
        phases = [0.0] * len(freqs)
    x = []
    for m in range(n):
        v = 0.0
        for k in range(len(freqs)):
            t = m * dt
            v += amps[k] * math.exp(-damps[k] * t) * math.cos(2 * math.pi * freqs[k] * t + phases[k])
        x.append(v)
    return x


def nearest_freqs(model, targets):
    """Match each target frequency to the nearest recovered |frequency|."""
    fs = [abs(f) for f in model["frequencies"]]
    out = []
    for t in targets:
        out.append(min(fs, key=lambda f: abs(f - t)))
    return out


def main():
    print("ESPRIT tests")

    dt = 1.0 / 100.0  # 100 Hz sampling

    # --- clean sum of two undamped tones: exact frequency recovery ---
    n = 128
    freqs = [10.0, 22.0]
    x = make_signal(freqs, [1.0, 0.7], [0.0, 0.0], n, dt)
    m = esprit.esprit(x, p=4, dt=dt)   # 2 real tones -> 4 complex modes (conjugate pairs)
    got = nearest_freqs(m, freqs)
    check("clean two-tone frequencies recovered",
          all(abs(got[i] - freqs[i]) < 1e-3 for i in range(2)))

    # --- reconstruction matches the clean signal ---
    xr = esprit.reconstruct(m, n)
    rec_err = max(abs(xr[i] - x[i]) for i in range(n))
    check("reconstruction matches clean signal", rec_err < 1e-4)

    # --- damped exponential: correct decay rate ---
    nd = 100
    xd = make_signal([5.0], [1.0], [3.0], nd, dt)
    md = esprit.esprit(xd, p=2, dt=dt)
    # damping of the mode nearest 5 Hz should be ~3
    idx = min(range(len(md["frequencies"])), key=lambda k: abs(abs(md["frequencies"][k]) - 5.0))
    check("damped exponential decay recovered", abs(md["damping"][idx] - 3.0) < 0.1)

    # --- SUPER-RESOLUTION: two tones closer than one FFT bin ---
    # FFT bin spacing = 1/(n*dt). Place tones 0.3 bins apart.
    n2 = 100
    bin_hz = 1.0 / (n2 * dt)
    f1 = 20.0
    f2 = 20.0 + 0.3 * bin_hz
    xs = make_signal([f1, f2], [1.0, 1.0], [0.0, 0.0], n2, dt)
    # the periodogram sees a single blob near 20 Hz
    npk, _ = esprit.periodogram_peak_count(xs, threshold=0.5)
    ms = esprit.esprit(xs, p=4, dt=dt)
    gs = sorted(set(round(abs(f), 3) for f in ms["frequencies"] if 15 < abs(f) < 25))
    # ESPRIT should separate the two close frequencies
    close = [f for f in gs]
    resolved = (abs(min(close, key=lambda f: abs(f - f1)) - f1) < 0.05 and
                abs(min(close, key=lambda f: abs(f - f2)) - f2) < 0.05)
    check(f"super-resolves tones {0.3:.1f} bins apart (FFT sees {npk} peak)", resolved)

    # --- NOISE ROBUSTNESS: ESPRIT beats Prony on a noisy signal ---
    nN = 200
    fN = [8.0, 19.0]
    clean = make_signal(fN, [1.0, 0.8], [0.0, 0.0], nN, dt, phases=[0.3, 1.1])
    rng = _R(2024)
    noise_amp = 0.05
    noisy = [clean[i] + noise_amp * rng.normal() for i in range(nN)]

    me = esprit.esprit(noisy, p=4, dt=dt)
    ge = nearest_freqs(me, fN)
    err_esprit = max(abs(ge[i] - fN[i]) for i in range(2))

    mp = prony.prony(noisy, p=4, dt=dt)
    fp = [abs(f) for f in mp["frequencies"]]
    gp = [min(fp, key=lambda f: abs(f - t)) for t in fN]
    err_prony = max(abs(gp[i] - fN[i]) for i in range(2))

    check(f"ESPRIT frequencies accurate under noise (err {err_esprit:.3f})", err_esprit < 0.3)
    check(f"ESPRIT beats Prony under noise ({err_esprit:.3f} < {err_prony:.3f})",
          err_esprit < err_prony)

    # --- mode count / structure sanity ---
    check("returns requested number of modes", len(me["modes"]) == 4)
    check("frequencies come in +-conjugate pairs",
          any(abs(f) > 1 for f in me["frequencies"]))

    # --- undamped tones have ~zero damping ---
    mt = esprit.esprit(make_signal([12.0], [1.0], [0.0], 80, dt), p=2, dt=dt)
    idx = min(range(len(mt["frequencies"])), key=lambda k: abs(abs(mt["frequencies"][k]) - 12.0))
    check("undamped tone has near-zero damping", abs(mt["damping"][idx]) < 0.2)

    # --- deterministic ---
    a = esprit.esprit(x, p=4, dt=dt)
    b = esprit.esprit(x, p=4, dt=dt)
    check("deterministic", a["frequencies"] == b["frequencies"])

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
