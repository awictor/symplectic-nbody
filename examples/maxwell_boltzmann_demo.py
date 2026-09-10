"""Demo: the Maxwell-Boltzmann speed distribution.

Prints the three characteristic speeds and their fixed ratios for several gases, then
draws the speed distributions with the most-probable, mean and rms speeds marked -- the
shape whose exp(-v^2) tail governs escape and fusion.

    python examples/maxwell_boltzmann_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from maxwell_boltzmann import (distribution, most_probable_speed,  # noqa: E402
                               mean_speed, rms_speed, mean_kinetic_energy,
                               fraction_above, AMU)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    T = 300.0
    print(f"Maxwell-Boltzmann speeds at T = {T:.0f} K "
          f"(mean KE = {mean_kinetic_energy(T)/1.602e-19:.3f} eV)\n")
    print(f"  v_p : <v> : v_rms = 1 : 1.128 : 1.225 (universal)\n")
    print(f"  {'gas':>8}{'mass (amu)':>12}{'v_p':>9}{'<v>':>9}{'v_rms':>9}"
          f"{'>3 v_p':>10}")
    print("  " + "-" * 57)
    gases = [("H2", 2.0), ("He", 4.0), ("N2", 28.0), ("O2", 32.0), ("CO2", 44.0)]
    for name, amu in gases:
        m = amu * AMU
        vp = most_probable_speed(T, m)
        frac = fraction_above(3 * vp, T, m)
        print(f"  {name:>8}{amu:>12.0f}{vp:>9.0f}{mean_speed(T, m):>9.0f}"
              f"{rms_speed(T, m):>9.0f}{frac:>10.1e}")

    print("\n  Speeds scale as 1/sqrt(m), so hydrogen zips along four times faster than")
    print("  nitrogen at the same temperature -- which is why light gases escape")
    print("  atmospheres and why sound (set by ~v_rms) travels faster in helium. The")
    print("  fraction above 3 v_p is only ~0.04%: the exp(-v^2) tail is thin, but it")
    print("  is exactly that tail that lets atoms escape gravity and nuclei fuse.")

    _svg(os.path.join(outdir, "maxwell_boltzmann.svg"), T)
    print(f"\n  wrote {os.path.join(outdir, 'maxwell_boltzmann.svg')}")


def _svg(path, T, size=720, pad=70):
    gases = [("H2", 2.0, "#4dabf7"), ("He", 4.0, "#06d6a0"),
             ("N2", 28.0, "#ffd43b"), ("CO2", 44.0, "#ff6b6b")]
    vmax = 3000.0
    vs = [vmax * i / 300 for i in range(1, 301)]
    curves = []
    fmax = 0.0
    for name, amu, col in gases:
        m = amu * AMU
        ys = [distribution(v, T, m) for v in vs]
        fmax = max(fmax, max(ys))
        curves.append((name, col, amu, ys))

    def sx(v):
        return pad + v / vmax * (size - 2 * pad)

    def sy(y):
        return size - pad - y / (fmax * 1.05) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    ytop = pad + 22
    for i, (name, col, amu, ys) in enumerate(curves):
        poly = " ".join(f"{sx(vs[j]):.1f},{sy(ys[j]):.1f}" for j in range(len(vs)))
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{col}" stroke-width="2.2"/>')
        parts.append(f'<text x="{size-pad-90}" y="{ytop + i*16}" fill="{col}" '
                     f'font-size="12">{name}</text>')

    # mark v_p, <v>, v_rms on the N2 curve
    m_n2 = 28.0 * AMU
    for label, v, col in [("v_p", most_probable_speed(T, m_n2), "#8b949e"),
                          ("<v>", mean_speed(T, m_n2), "#8b949e"),
                          ("v_rms", rms_speed(T, m_n2), "#8b949e")]:
        x = sx(v)
        parts.append(f'<line x1="{x:.1f}" y1="{sy(distribution(v, T, m_n2)):.1f}" '
                     f'x2="{x:.1f}" y2="{size-pad:.1f}" stroke="{col}" '
                     f'stroke-width="0.8" stroke-dasharray="3 3"/>')
        parts.append(f'<text x="{x:.1f}" y="{size-pad+14:.1f}" fill="{col}" '
                     f'font-size="9" text-anchor="middle">{label}</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Maxwell-Boltzmann speed distribution (300 K)</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'lighter gases peak faster (v ~ 1/sqrt(m)); ticks on N2</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+28:.1f}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">molecular speed (m/s) -&gt;</text>')
    parts.append(f'<text x="{pad-16}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'probability density f(v)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
