"""Demo: thin-film interference -- soap-bubble colours and Newton's rings.

Prints the reflected bright colour of a soap film as it thins, the MgF2 anti-reflection
coating for a lens, then draws two figures: the bright-reflection wavelength versus film
thickness, and the concentric Newton's rings in the air gap under a lens.

    python examples/thin_film_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from thin_film import (optical_path, constructive_wavelength,  # noqa: E402
                       antireflection_thickness, ideal_ar_index, newton_ring_radius)


N_SOAP = 1.33


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Thin-film interference: a film nm-thick paints itself in colour\n")
    print("  Soap film (n = 1.33) -- first-order bright reflected wavelength:")
    print(f"  {'thickness':>12}{'2 n t':>12}{'bright lambda':>16}{'colour':>12}")
    for t in (80e-9, 110e-9, 150e-9, 200e-9, 250e-9):
        lam = constructive_wavelength(N_SOAP, t, 1)
        print(f"  {t*1e9:>9.0f} nm{optical_path(N_SOAP,t)*1e9:>9.0f} nm{lam*1e9:>13.0f} nm{_colour(lam):>12}")

    print("\n  Anti-reflection coating for a glass lens (n_glass = 1.52) at 550 nm:")
    print("    ideal index = sqrt(1.52) = %.3f (MgF2 at 1.38 is the practical pick)" % ideal_ar_index(1.52))
    print("    quarter-wave MgF2 thickness = %.0f nm" % (antireflection_thickness(550e-9, 1.38) * 1e9))
    print("    -> the two reflections cancel, killing glare and boosting transmission.")

    print("\n  A soap film thins to near-zero before it bursts: 2 n t -> 0 is destructive at")
    print("  every colour, so the film looks BLACK -- the classic sign it is about to pop.")

    _svg(os.path.join(outdir, "thin_film.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'thin_film.svg')}")


def _colour(lam):
    nm = lam * 1e9
    if nm < 440:
        return "violet"
    if nm < 490:
        return "blue"
    if nm < 560:
        return "green"
    if nm < 590:
        return "yellow"
    if nm < 640:
        return "orange"
    if nm < 750:
        return "red"
    return "infrared"


def _wl_to_rgb(nm):
    # rough visible-spectrum colour for the SVG
    if nm < 440:
        return "#8338ec"
    if nm < 490:
        return "#4dabf7"
    if nm < 560:
        return "#06d6a0"
    if nm < 590:
        return "#ffd43b"
    if nm < 640:
        return "#ff922b"
    if nm < 750:
        return "#ff6b6b"
    return "#7a2020"


def _svg(path, size=720, pad=70):
    # Left: bright wavelength vs film thickness. Right: Newton's rings.
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">Thin-film interference</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'soap-film bright colour vs thickness (left); Newton&#39;s rings under a lens (right)</text>',
    ]

    # --- left panel: bright wavelength vs thickness ---
    lx0, lx1 = pad, size * 0.52
    ly0, ly1 = size - pad, pad + 44
    t_min, t_max = 60e-9, 320e-9
    lam_min, lam_max = 380e-9, 760e-9

    def TX(t):
        return lx0 + (t - t_min) / (t_max - t_min) * (lx1 - lx0)

    def LY(lam):
        return ly0 - (lam - lam_min) / (lam_max - lam_min) * (ly0 - ly1)

    # visible-band shading on the y axis
    for nm in range(400, 760, 8):
        parts.append(f'<rect x="{lx0-16:.1f}" y="{LY(nm*1e-9)-3:.1f}" width="12" height="5" '
                     f'fill="{_wl_to_rgb(nm)}" opacity="0.8"/>')

    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx1}" y2="{ly0}" stroke="#8b949e" stroke-width="1.4"/>')
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx0}" y2="{ly1}" stroke="#8b949e" stroke-width="1.4"/>')
    # bright curve (first order): lambda = 2 n t / 0.5 = 4 n t, clipped to visible
    pts = []
    n = 120
    for i in range(n + 1):
        t = t_min + (t_max - t_min) * i / n
        lam = constructive_wavelength(N_SOAP, t, 1)
        if lam_min <= lam <= lam_max:
            pts.append(f"{TX(t):.1f},{LY(lam):.1f}")
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#e6edf3" stroke-width="2.4"/>')
    # colour dots at sample thicknesses
    for t in (90e-9, 120e-9, 150e-9, 180e-9):
        lam = constructive_wavelength(N_SOAP, t, 1)
        if lam_min <= lam <= lam_max:
            parts.append(f'<circle cx="{TX(t):.1f}" cy="{LY(lam):.1f}" r="5" fill="{_wl_to_rgb(lam*1e9)}"/>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{ly0+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">film thickness (60-320 nm)</text>')
    parts.append(f'<text x="{lx0-22:.1f}" y="{(ly0+ly1)/2:.1f}" fill="#8b949e" font-size="10" '
                 f'transform="rotate(-90 {lx0-22:.1f} {(ly0+ly1)/2:.1f})" text-anchor="middle">bright wavelength</text>')

    # --- right panel: Newton's rings ---
    rcx = size * 0.78
    rcy = size * 0.52
    R = 2.0
    lam = 550e-9
    # scale so ~6 rings fit
    r6 = newton_ring_radius(6, lam, R)
    scale = (size * 0.20) / r6
    parts.append(f'<circle cx="{rcx:.1f}" cy="{rcy:.1f}" r="{size*0.21:.1f}" fill="#111820"/>')
    for m in range(1, 7):
        rd = newton_ring_radius(m, lam, R) * scale
        parts.append(f'<circle cx="{rcx:.1f}" cy="{rcy:.1f}" r="{rd:.1f}" fill="none" '
                     f'stroke="#4dabf7" stroke-width="2" opacity="0.85"/>')
    parts.append(f'<circle cx="{rcx:.1f}" cy="{rcy:.1f}" r="2.5" fill="#8b949e"/>')
    parts.append(f'<text x="{rcx:.1f}" y="{rcy+size*0.24:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">dark rings r = sqrt(m lambda R)</text>')
    parts.append(f'<text x="{rcx:.1f}" y="{rcy-size*0.23:.1f}" fill="#8b949e" font-size="11" '
                 f'text-anchor="middle">Newton&#39;s rings</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
