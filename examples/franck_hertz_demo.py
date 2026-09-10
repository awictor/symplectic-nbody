"""Demo: the Franck-Hertz experiment -- atomic energy levels in a current curve.

Prints the dip voltages, excitation counts and emission wavelength for mercury, then draws
the signature Franck-Hertz current-versus-voltage sawtooth: current climbs, then drops every
time electrons gain one more quantum of excitation energy (4.9 V for mercury).

    python examples/franck_hertz_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from franck_hertz import (dip_voltages, dip_spacing, num_excitations,  # noqa: E402
                          emission_wavelength, residual_energy, HG_EXCITATION_EV)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    ex = HG_EXCITATION_EV
    print("Franck-Hertz: current dips prove quantized atomic energy (mercury, %.1f eV)\n" % ex)
    print("  Current dips at multiples of the excitation voltage:")
    for i, v in enumerate(dip_voltages(ex, 5), 1):
        print(f"    dip {i}:  {v:>5.1f} V   (electron has excited the atom {i} time%s)"
              % ("" if i == 1 else "s"))

    print("\n  Dip spacing = %.1f V = the mercury 6s6p excitation energy in volts." % dip_spacing(ex))
    print("  Emission on relaxation: lambda = h c / E = %.0f nm (the mercury UV line).\n"
          % (emission_wavelength(ex) * 1e9))

    print(f"  {'accel V':>10}{'excitations':>14}{'residual (eV)':>16}")
    for V in (3, 6, 11, 16, 25):
        print(f"  {V:>8} V{num_excitations(V, ex):>14}{residual_energy(V, ex):>16.2f}")

    print("\n  Electrons collide elastically (no energy lost) until they reach 4.9 eV, then can")
    print("  dump exactly that quantum inelastically and arrive too slow to be collected -- so")
    print("  the current drops. Repeating dips = repeated excitations = energy is quantized.")

    _svg(os.path.join(outdir, "franck_hertz.svg"), ex)
    print(f"\n  wrote {os.path.join(outdir, 'franck_hertz.svg')}")


def _svg(path, ex, size=720, pad=76):
    v_max = 30.0
    x0, x1 = pad, size - pad
    y0, y1 = size - pad, pad + 44

    def X(v):
        return x0 + v / v_max * (x1 - x0)

    # model current: rising ramp with a sharp drop each time V passes a multiple of ex
    def current(v):
        # base rises with sqrt(v); subtract a dip near each n*ex
        base = math.sqrt(max(v, 0.0))
        dip = 0.0
        n = 1
        while n * ex <= v_max:
            centre = n * ex + 0.3     # dips sit just past the multiple (contact potential-ish)
            dip += 0.8 * math.exp(-((v - centre) / 0.7) ** 2)
            n += 1
        return max(0.0, base - dip)

    vs = [v_max * i / 400 for i in range(401)]
    cs = [current(v) for v in vs]
    cmax = max(cs) * 1.1

    def Y(c):
        return y0 - c / cmax * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">The Franck-Hertz current curve</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'current drops every 4.9 V -- each drop is electrons dumping one quantum into a mercury atom</text>',
    ]

    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#8b949e" stroke-width="1.5"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#8b949e" stroke-width="1.5"/>')

    # dip voltage gridlines
    for v in dip_voltages(ex, 6):
        if v <= v_max:
            parts.append(f'<line x1="{X(v):.1f}" y1="{y0:.1f}" x2="{X(v):.1f}" y2="{y1:.1f}" '
                         f'stroke="#ff6b6b" stroke-width="0.8" stroke-dasharray="4 4" opacity="0.5"/>')
            parts.append(f'<text x="{X(v):.1f}" y="{y0+15:.1f}" fill="#ff6b6b" font-size="9" '
                         f'text-anchor="middle">{v:.1f}</text>')
    # voltage axis ticks
    for v in range(0, 31, 5):
        parts.append(f'<text x="{X(v):.1f}" y="{y0+28:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{v} V</text>')

    poly = " ".join(f"{X(vs[i]):.1f},{Y(cs[i]):.1f}" for i in range(len(vs)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#4dabf7" stroke-width="2.6"/>')

    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{y0+42:.1f}" fill="#8b949e" font-size="11" '
                 f'text-anchor="middle">accelerating voltage (red lines = 4.9 V dip spacing)</text>')
    parts.append(f'<text x="{x0-24:.1f}" y="{(y0+y1)/2:.1f}" fill="#8b949e" font-size="11" '
                 f'transform="rotate(-90 {x0-24:.1f} {(y0+y1)/2:.1f})" text-anchor="middle">collector current</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
