"""Demo: the photoelectric effect.

Prints the threshold wavelength and the stopping voltage under a few light sources for
several metals, then draws the stopping voltage vs frequency -- parallel lines whose
common slope is Planck's constant h and whose x-intercepts are the work functions.

    python examples/photoelectric_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from photoelectric import (threshold_wavelength, threshold_frequency,  # noqa: E402
                           stopping_voltage, max_kinetic_energy, is_emitting,
                           PHI_CESIUM, PHI_SODIUM, PHI_ZINC, PHI_PLATINUM,
                           C, H, EV, E_CHARGE)

METALS = [("cesium", PHI_CESIUM), ("sodium", PHI_SODIUM),
          ("zinc", PHI_ZINC), ("platinum", PHI_PLATINUM)]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Photoelectric effect: K_max = h f - phi (Einstein 1905)\n")
    print(f"  slope of V_stop vs f is h/e = {H/E_CHARGE:.3e} V/Hz (universal)\n")
    print(f"  {'metal':>10}{'phi (eV)':>10}{'threshold':>12}{'V_stop @254nm':>15}"
          f"{'V_stop @400nm':>15}")
    print("  " + "-" * 62)
    f254 = C / 254e-9
    f400 = C / 400e-9
    for name, phi in METALS:
        lam0 = threshold_wavelength(phi) * 1e9
        v254 = stopping_voltage(f254, phi)
        v400 = stopping_voltage(f400, phi)
        s254 = f"{v254:.2f} V" if is_emitting(f254, phi) else "none"
        s400 = f"{v400:.2f} V" if is_emitting(f400, phi) else "none"
        print(f"  {name:>10}{phi:>10.2f}{lam0:>10.0f}nm{s254:>15}{s400:>15}")

    print("\n  Below the threshold frequency, no electrons come off however bright the")
    print("  light -- energy comes in photon lumps, not a continuous wave. Above it, the")
    print("  stopping voltage climbs linearly with frequency at the SAME slope h/e for")
    print("  every metal; only the intercept (the work function) differs. Millikan")
    print("  measured exactly this line and pinned down Planck's constant.")

    _svg(os.path.join(outdir, "photoelectric.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'photoelectric.svg')}")


def _svg(path, size=720, pad=72):
    freqs = [0.2e15 * i for i in range(1, 36)]   # up to ~7e15 Hz
    xmin, xmax = freqs[0] / 1e15, freqs[-1] / 1e15
    # y range from stopping voltages
    vmax = max(stopping_voltage(f, PHI_CESIUM) for f in freqs)

    def sx(x):
        return pad + (x - xmin) / (xmax - xmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - y / (vmax * 1.05) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    cols = ["#4dabf7", "#ffd43b", "#06d6a0", "#ff6b6b"]
    ytop = pad + 22
    for i, (name, phi) in enumerate(METALS):
        pts = []
        for f in freqs:
            v = stopping_voltage(f, phi)
            if v > 0:
                pts.append(f"{sx(f/1e15):.1f},{sy(v):.1f}")
        if pts:
            parts.append(f'<polyline points="{" ".join(pts)}" fill="none" '
                         f'stroke="{cols[i]}" stroke-width="2.2"/>')
        # threshold tick on the x-axis
        f0 = threshold_frequency(phi) / 1e15
        parts.append(f'<circle cx="{sx(f0):.1f}" cy="{sy(0):.1f}" r="3" fill="{cols[i]}"/>')
        parts.append(f'<text x="{pad+10}" y="{ytop + i*16}" fill="{cols[i]}" '
                     f'font-size="12">{name} (phi={phi:.1f} eV)</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Stopping voltage vs frequency</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'parallel lines: common slope h/e, intercepts = work functions</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">frequency (10^15 Hz) -&gt;</text>')
    parts.append(f'<text x="{pad-18}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'stopping voltage (V)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
