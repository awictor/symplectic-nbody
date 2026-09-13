"""Demo: Hilbert transform -- envelope detection and instantaneous frequency.

Recovers the amplitude envelope of an AM signal and the rising frequency of a chirp using the analytic
signal. Draws the AM signal with its recovered envelope and the chirp's instantaneous frequency.

    python examples/hilbert_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hilbert import envelope, instantaneous_frequency, hilbert_transform  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Hilbert transform: envelope and instantaneous frequency from the analytic signal\n")

    # AM signal: a slow envelope riding a fast carrier
    N = 256
    t = [i / N for i in range(N)]
    mod = [1 + 0.6 * math.cos(2 * math.pi * 2 * x) for x in t]
    am = [mod[i] * math.cos(2 * math.pi * 25 * t[i]) for i in range(N)]
    env = envelope(am)
    err = max(abs(env[i] - mod[i]) for i in range(20, N - 20))
    print("  Amplitude modulation: a 25 Hz carrier under a 2 Hz envelope.")
    print(f"    recovered envelope matches the true modulating amplitude to {err:.4f}\n")

    # FM chirp: frequency sweeps from 5 to ~42 Hz, staying below the 50 Hz Nyquist limit
    N = 300
    fs = 100.0
    f0, r = 5.0, 25.0
    chirp = [math.cos(2 * math.pi * (f0 * (i / fs) + 0.5 * r * (i / fs) ** 2)) for i in range(N)]
    ifreq = instantaneous_frequency(chirp, sample_rate=fs)
    print("  Linear chirp: instantaneous frequency should rise from 5 Hz to ~42 Hz (below Nyquist).")
    print(f"    {'time (s)':>9}  {'true freq':>10}  {'measured':>9}")
    for ti in [0.3, 0.8, 1.2, 1.6]:
        idx = int(ti * fs)
        if idx < len(ifreq):
            print(f"    {ti:>9.1f}  {f0 + r*ti:>10.2f}  {ifreq[idx]:>9.2f}")

    print("\n  The analytic signal z = x + i H[x] = A e^{i phi} hands you both: the envelope A(t)")
    print("  and the phase phi(t) whose derivative is the instantaneous frequency. AM and FM")
    print("  demodulation, vibration diagnostics, and empirical mode decomposition all build on it.")

    _svg(os.path.join(outdir, "hilbert.svg"), t, am, env, mod, ifreq, fs, f0, r)
    print(f"\n  wrote {os.path.join(outdir, 'hilbert.svg')}")


def _svg(path, t, am, env, mod, ifreq, fs, f0, r, width=760, height=420):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="24" fill="#e6edf3" font-size="15">'
        'Top: AM signal (gray) with recovered envelope (green). Bottom: chirp instantaneous freq</text>',
    ]

    # ---- top: AM + envelope -------------------------------------------------------------
    ox, oy, ow, oh = 40, 45, width - 80, 150
    N = len(am)
    amax = max(max(am), max(env))

    def sx(i):
        return ox + ow * i / (N - 1)

    def sy(v):
        return oy + oh * (1 - (v + amax) / (2 * amax))

    parts.append(f'<line x1="{ox}" y1="{sy(0):.1f}" x2="{ox+ow}" y2="{sy(0):.1f}" '
                 f'stroke="#30363d" stroke-width="0.5"/>')
    amp = " ".join(f"{sx(i):.1f},{sy(am[i]):.1f}" for i in range(N))
    parts.append(f'<polyline points="{amp}" fill="none" stroke="#8b949e" stroke-width="0.8"/>')
    ep = " ".join(f"{sx(i):.1f},{sy(env[i]):.1f}" for i in range(N))
    parts.append(f'<polyline points="{ep}" fill="none" stroke="#06d6a0" stroke-width="2"/>')
    en = " ".join(f"{sx(i):.1f},{sy(-env[i]):.1f}" for i in range(N))
    parts.append(f'<polyline points="{en}" fill="none" stroke="#06d6a0" stroke-width="2"/>')

    # ---- bottom: instantaneous frequency ------------------------------------------------
    bx, by, bw, bh = 40, 240, width - 80, 150
    M = len(ifreq)
    fmax = max(ifreq)

    def fx(i):
        return bx + bw * i / (M - 1)

    def fy(v):
        return by + bh * (1 - v / fmax)

    parts.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" fill="none" stroke="#30363d"/>')
    # true line f0 + r*t
    true_pts = " ".join(f"{fx(i):.1f},{fy(f0 + r*(i/fs)):.1f}" for i in range(M))
    parts.append(f'<polyline points="{true_pts}" fill="none" stroke="#ff6b6b" stroke-width="1.5" '
                 f'stroke-dasharray="5 3"/>')
    meas = " ".join(f"{fx(i):.1f},{fy(ifreq[i]):.1f}" for i in range(M))
    parts.append(f'<polyline points="{meas}" fill="none" stroke="#ffd43b" stroke-width="1.3"/>')
    parts.append(f'<text x="{bx+bw-150}" y="{by+16}" fill="#ff6b6b" font-size="10">true freq</text>')
    parts.append(f'<text x="{bx+bw-150}" y="{by+31}" fill="#ffd43b" font-size="10">measured (Hilbert)</text>')
    parts.append(f'<text x="{bx+bw/2:.0f}" y="{by+bh+18:.0f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">time -- chirp frequency rising</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
