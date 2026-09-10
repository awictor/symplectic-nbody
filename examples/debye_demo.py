"""Demo: Debye shielding and the plasma frequency.

Prints the Debye length, plasma-sphere population and plasma frequency across plasma
environments, then draws the plasma frequency vs electron density with the AM/FM radio
bands marked -- the cutoff that bounces AM around the Earth but lets FM escape.

    python examples/debye_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from debye import (debye_length, plasma_parameter, plasma_frequency_hz,  # noqa: E402
                   critical_density)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Debye shielding: lambda_D = sqrt(eps0 kT / n e^2); plasma if N_D >> 1\n")
    print(f"  {'environment':>20}{'n (/m^3)':>11}{'T (K)':>9}"
          f"{'lambda_D':>12}{'N_D':>10}{'f_p':>11}")
    print("  " + "-" * 73)
    envs = [
        ("solar wind (1 AU)", 5e6, 1e5),
        ("ionosphere", 1e12, 1e3),
        ("solar corona", 1e15, 2e6),
        ("lab (tokamak edge)", 1e18, 1e5),
        ("fusion core", 1e20, 1e8),
    ]
    for name, n, T in envs:
        lD = debye_length(n, T)
        ND = plasma_parameter(n, T)
        fp = plasma_frequency_hz(n)
        lstr = f"{lD*1e3:.2f} mm" if lD < 1 else f"{lD:.1f} m"
        if fp < 1e6:
            fstr = f"{fp/1e3:.0f} kHz"
        elif fp < 1e9:
            fstr = f"{fp/1e6:.1f} MHz"
        else:
            fstr = f"{fp/1e9:.1f} GHz"
        print(f"  {name:>20}{n:>11.0e}{T:>9.0e}{lstr:>12}{ND:>10.1e}{fstr:>11}")

    print("\n  Below the plasma frequency, EM waves are reflected. The ionosphere's")
    print("  ~9 MHz cutoff is why AM radio (~1 MHz) bounces around the curve of the")
    print("  Earth while FM (~100 MHz) and TV punch straight through to space. The")
    print("  Debye length is the screening radius, and with thousands of particles")
    print("  inside a Debye sphere the shielding is a smooth collective effect.")

    _svg(os.path.join(outdir, "debye.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'debye.svg')}")


def _svg(path, size=720, pad=70):
    ns = [10 ** (5 + 0.2 * i) for i in range(0, 81)]   # 1e5 .. 1e21
    fps = [plasma_frequency_hz(n) for n in ns]
    lx = [math.log10(n) for n in ns]
    ly = [math.log10(f) for f in fps]
    xmin, xmax = lx[0], lx[-1]
    ymin, ymax = min(ly), max(ly)

    def sx(x):
        return pad + (x - xmin) / (xmax - xmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - ymin) / (ymax - ymin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    # radio-band horizontal lines
    for f_hz, label, col in [(1e6, "AM ~1 MHz", "#ff6b6b"),
                             (1e8, "FM ~100 MHz", "#06d6a0"),
                             (2.4e9, "WiFi 2.4 GHz", "#b197fc")]:
        if ymin <= math.log10(f_hz) <= ymax:
            yb = sy(math.log10(f_hz))
            parts.append(f'<line x1="{pad}" y1="{yb:.1f}" x2="{size-pad}" y2="{yb:.1f}" '
                         f'stroke="{col}" stroke-width="1" stroke-dasharray="3 4" opacity="0.7"/>')
            parts.append(f'<text x="{pad+8}" y="{yb-5:.1f}" fill="{col}" '
                         f'font-size="10">{label}</text>')

    poly = " ".join(f"{sx(lx[i]):.1f},{sy(ly[i]):.1f}" for i in range(len(ns)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#ffd43b" stroke-width="2.6"/>')

    for name, n, col in [("ionosphere", 1e12, "#4dabf7"),
                         ("corona", 1e15, "#ff922b"),
                         ("fusion core", 1e20, "#f06595")]:
        fp = plasma_frequency_hz(n)
        px = sx(math.log10(n))
        py = sy(math.log10(fp))
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="{col}"/>')
        parts.append(f'<text x="{px-8:.1f}" y="{py-6:.1f}" fill="{col}" '
                     f'font-size="10" text-anchor="end">{name}</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Plasma frequency vs electron density</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'waves below f_p are reflected: the ionosphere bounces AM, passes FM</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 electron density (m^-3) -&gt;</text>')
    parts.append(f'<text x="{pad-16}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 plasma frequency (Hz)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
