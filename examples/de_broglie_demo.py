"""Demo: the de Broglie wavelength of matter.

Prints matter wavelengths from electrons to a baseball, then draws the de Broglie
wavelength vs kinetic energy for electron, proton and neutron, with visible-light and
atomic-spacing reference lines showing where each probe resolves structure.

    python examples/de_broglie_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from de_broglie import (wavelength_from_energy, wavelength_from_velocity,  # noqa: E402
                        thermal_wavelength, electron_microscope_wavelength,
                        M_E, M_P, M_N, EV)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("de Broglie wavelength: lambda = h / p -- everything is a wave\n")
    print(f"  {'object':>26}{'lambda':>16}")
    print("  " + "-" * 44)
    rows = [
        ("100 keV microscope e-", electron_microscope_wavelength(1e5)),
        ("1 eV electron", wavelength_from_energy(EV, M_E)),
        ("thermal neutron (300 K)", thermal_wavelength(300.0, M_N)),
        ("thermal He atom (300 K)", thermal_wavelength(300.0, 4 * M_P)),
        ("100 m/s N2 molecule", wavelength_from_velocity(100.0, 28 * M_P)),
        ("baseball (40 m/s)", wavelength_from_velocity(40.0, 0.145)),
    ]
    for name, lam in rows:
        if lam > 1e-9:
            s = f"{lam*1e9:.3g} nm"
        elif lam > 1e-12:
            s = f"{lam*1e12:.3g} pm"
        else:
            s = f"{lam:.2e} m"
        print(f"  {name:>26}{s:>16}")

    print("\n  Momentum sets the wavelength, so heavy or fast things have vanishingly")
    print("  short waves -- a baseball's is 10^-34 m, undetectable. But a 100 keV")
    print("  electron's 4 pm is thousands of times finer than visible light, which is")
    print("  why electron microscopes resolve atoms, and a thermal neutron's ~0.1 nm")
    print("  matches crystal spacing, making neutron diffraction a structural probe.")

    _svg(os.path.join(outdir, "de_broglie.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'de_broglie.svg')}")


def _svg(path, size=720, pad=72):
    Es = [10 ** (-2 + 0.08 * i) for i in range(0, 101)]   # 0.01 eV .. ~1e6 eV
    particles = [("electron", M_E, "#4dabf7"), ("proton", M_P, "#ffd43b"),
                 ("neutron", M_N, "#ff6b6b")]
    data = []
    for name, m, col in particles:
        ys = [wavelength_from_energy(E * EV, m) for E in Es]
        data.append((name, col, ys))

    lx = [math.log10(E) for E in Es]
    all_y = [y for _, _, ys in data for y in ys]
    ly_all = [math.log10(y) for y in all_y]
    xmin, xmax = lx[0], lx[-1]
    ymin, ymax = min(ly_all), max(ly_all)

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
    # reference lines: visible light ~500 nm, atomic spacing ~0.2 nm
    for lam_ref, label, col in [(5e-7, "visible light ~500 nm", "#8b949e"),
                                (2e-10, "atomic spacing ~0.2 nm", "#06d6a0")]:
        if ymin <= math.log10(lam_ref) <= ymax:
            yr = sy(math.log10(lam_ref))
            parts.append(f'<line x1="{pad}" y1="{yr:.1f}" x2="{size-pad}" y2="{yr:.1f}" '
                         f'stroke="{col}" stroke-width="1" stroke-dasharray="3 4" opacity="0.7"/>')
            parts.append(f'<text x="{pad+8}" y="{yr-5:.1f}" fill="{col}" '
                         f'font-size="10">{label}</text>')

    ytop = pad + 22
    for i, (name, col, ys) in enumerate(data):
        ly = [math.log10(y) for y in ys]
        poly = " ".join(f"{sx(lx[j]):.1f},{sy(ly[j]):.1f}" for j in range(len(Es)))
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{col}" stroke-width="2.4"/>')
        parts.append(f'<text x="{size-pad-90}" y="{ytop + i*16}" fill="{col}" '
                     f'font-size="12">{name}</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'de Broglie wavelength vs kinetic energy</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'faster/heavier = shorter wave; below atomic spacing = resolves atoms</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 kinetic energy (eV) -&gt;</text>')
    parts.append(f'<text x="{pad-18}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 wavelength (m)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
