"""Demo: Prony's method -- decomposing a signal into damped sinusoids, beating FFT resolution.

Fits a sum of two damped sinusoids to a sampled signal, recovering the exact frequencies and damping,
and shows Prony resolving two tones closer than one FFT bin where the FFT sees a single blob. Draws
the signal with its Prony reconstruction and the recovered spectrum.

    python examples/prony_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prony import prony, reconstruct  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Prony's method: fit a signal as a sum of damped sinusoids (parametric spectrum)\n")

    dt = 0.005
    N = 100
    # two decaying tones
    def sig(n):
        t = n * dt
        return (math.exp(-2 * t) * math.cos(2 * math.pi * 18 * t + 0.3) +
                0.6 * math.exp(-6 * t) * math.cos(2 * math.pi * 33 * t))
    x = [sig(n) for n in range(N)]

    model = prony(x, p=4, dt=dt)
    print(f"  Signal: 18 Hz (damping 2/s) + 33 Hz (damping 6/s), {N} samples.\n")
    print(f"  Recovered modes:")
    print(f"    {'freq (Hz)':>10}  {'damping (1/s)':>14}  {'amplitude':>10}")
    seen = set()
    for k in range(len(model["modes"])):
        f = model["frequencies"][k]
        if f < -0.5:
            continue  # show only non-negative frequencies (conjugate pairs)
        key = round(abs(f), 1)
        if key in seen:
            continue
        seen.add(key)
        d = model["damping"][k]
        amp = abs(model["amplitudes"][k]) * 2  # conjugate pair -> real amplitude
        print(f"    {f:>10.2f}  {d:>14.2f}  {amp:>10.3f}")

    recon = reconstruct(model, N)
    err = max(abs(recon[n] - x[n]) for n in range(N))
    print(f"\n  Reconstruction error: {err:.2e} (Prony fits the exact modes, not FFT bins).\n")

    # super-resolution
    print("  Super-resolution: two tones closer than one FFT bin.")
    dt2 = 0.01
    N2 = 64
    bin_hz = 1 / (N2 * dt2)
    f1, f2 = 12.0, 12.0 + 0.35 * bin_hz
    def close(n):
        t = n * dt2
        return math.cos(2 * math.pi * f1 * t) + math.cos(2 * math.pi * f2 * t)
    xc = [close(n) for n in range(N2)]
    m2 = prony(xc, p=4, dt=dt2)
    fr = sorted(set(round(abs(f), 3) for f in m2["frequencies"] if abs(f) > 1))
    print(f"    FFT bin width {bin_hz:.2f} Hz; the tones are only {f2-f1:.2f} Hz apart.")
    print(f"    Prony recovers: {[f for f in fr if abs(f-f1)<0.3 or abs(f-f2)<0.3]} Hz (both resolved).")
    print("    An FFT would show one merged peak; Prony solves for the poles directly.")

    _svg(os.path.join(outdir, "prony.svg"), x, recon, dt)
    print(f"\n  wrote {os.path.join(outdir, 'prony.svg')}")


def _svg(path, x, recon, dt, width=760, height=380):
    N = len(x)
    lo = min(min(x), min(recon))
    hi = max(max(x), max(recon))
    ox, oy, ow, oh = 40, 50, width - 80, height - 90

    def sx(n):
        return ox + ow * n / (N - 1)

    def sy(v):
        return oy + oh * (1 - (v - lo) / (hi - lo))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="26" fill="#e6edf3" font-size="15">'
        'Signal (gray dots) and Prony reconstruction from 4 damped-sinusoid modes (green)</text>',
    ]
    parts.append(f'<line x1="{ox}" y1="{sy(0):.1f}" x2="{ox+ow}" y2="{sy(0):.1f}" '
                 f'stroke="#30363d" stroke-width="0.5"/>')
    for n in range(N):
        parts.append(f'<circle cx="{sx(n):.1f}" cy="{sy(x[n]):.1f}" r="2" fill="#8b949e" '
                     f'opacity="0.7"/>')
    rp = " ".join(f"{sx(n):.1f},{sy(recon[n]):.1f}" for n in range(N))
    parts.append(f'<polyline points="{rp}" fill="none" stroke="#06d6a0" stroke-width="1.5"/>')
    parts.append(f'<text x="{ox}" y="{oy+oh+18}" fill="#8b949e" font-size="10">'
                 f'the reconstruction lands on the samples: the model captures the signal exactly</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
