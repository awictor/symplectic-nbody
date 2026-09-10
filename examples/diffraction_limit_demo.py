"""Demo: the diffraction limit -- the resolution floor of every aperture.

Prints the angular resolution of real instruments from the human eye to a radio dish, then
draws resolution versus aperture on log-log axes with those instruments marked, plus a sketch
of two point sources at the Rayleigh limit (Airy disks just separable).

    python examples/diffraction_limit_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diffraction_limit import (rayleigh_angle, rayleigh_angle_arcsec,  # noqa: E402
                               abbe_limit, grating_resolving_power)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Diffraction limit: theta = 1.22 lambda / D -- aperture sets resolution\n")
    print(f"  {'instrument':<26}{'aperture':>12}{'lambda':>10}{'resolution':>16}")
    # (name, D, lambda)
    insts = [
        ("human eye", 2e-3, 550e-9),
        ("binoculars (50 mm)", 0.05, 550e-9),
        ("amateur scope (200 mm)", 0.2, 550e-9),
        ("Hubble (2.4 m)", 2.4, 550e-9),
        ("VLT (8.2 m)", 8.2, 550e-9),
        ("Arecibo radio (305 m)", 305.0, 0.21),
    ]
    for name, D, lam in insts:
        arcsec = rayleigh_angle_arcsec(lam, D)
        if arcsec > 60:
            rs = f"{arcsec/60:.1f} arcmin"
        else:
            rs = f"{arcsec:.3f} arcsec"
        ls = f"{lam*1e9:.0f} nm" if lam < 1e-6 else f"{lam*100:.0f} cm"
        print(f"  {name:<26}{D*1000 if D<1 else D:>9.0f}{'mm' if D<1 else ' m'}{ls:>10}{rs:>16}")

    print("\n  Microscope (Abbe d = lambda/2NA):")
    print(f"    light, NA 1.4:   {abbe_limit(550e-9, 1.4)*1e9:.0f} nm")
    print(f"    electron, 4 pm:  {abbe_limit(4e-12, 0.02)*1e12:.1f} pm (resolves atoms)")

    R = grating_resolving_power(1, 10000)
    print("\n  A 10000-line grating resolves lambda/dlambda = %.0f -- enough to split the" % R)
    print("  sodium doublet (589.0 vs 589.6 nm). Bigger apertures and more grating lines are")
    print("  the only way past the wave-optics floor; it's why telescopes and dishes grow huge.")

    _svg(os.path.join(outdir, "diffraction_limit.svg"), insts)
    print(f"\n  wrote {os.path.join(outdir, 'diffraction_limit.svg')}")


def _svg(path, insts, size=720, pad=80):
    # Top: log-log resolution (arcsec) vs aperture (m). Bottom: two Airy disks at Rayleigh sep.
    x0, x1 = pad, size - pad
    mid = size * 0.56

    # --- curve ---
    lam = 550e-9
    Ds = [10 ** (-3 + (math.log10(400) - (-3)) * i / 199) for i in range(200)]
    arc = [rayleigh_angle_arcsec(lam, D) for D in Ds]
    lx0, lx1 = math.log10(Ds[0]), math.log10(Ds[-1])
    ly0, ly1 = math.log10(min(arc)), math.log10(max(arc))
    ty0, ty1 = mid - 26, pad + 44

    def X(D):
        return x0 + (math.log10(D) - lx0) / (lx1 - lx0) * (x1 - x0)

    def Y(a):
        return ty0 - (math.log10(a) - ly0) / (ly1 - ly0) * (ty0 - ty1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">The diffraction limit</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'angular resolution (550 nm light) improves as 1/aperture -- why telescopes grow</text>',
    ]
    parts.append(f'<line x1="{x0}" y1="{ty0}" x2="{x1}" y2="{ty0}" stroke="#8b949e" stroke-width="1.4"/>')
    parts.append(f'<line x1="{x0}" y1="{ty0}" x2="{x0}" y2="{ty1}" stroke="#8b949e" stroke-width="1.4"/>')
    for e in range(-3, 3):
        gx = X(10.0 ** e)
        if x0 <= gx <= x1:
            lbl = f"{10**e:g} m" if e >= 0 else f"{10**(e+3):g} mm"
            parts.append(f'<text x="{gx:.1f}" y="{ty0+14:.1f}" fill="#8b949e" font-size="9" '
                         f'text-anchor="middle">{lbl}</text>')
    for e in range(int(math.floor(ly0)), int(math.ceil(ly1)) + 1):
        gy = Y(10.0 ** e)
        if ty1 <= gy <= ty0:
            lbl = f"{10**e:g}\"" if e >= 0 else f"{10**e:g}\""
            parts.append(f'<text x="{x0-6:.1f}" y="{gy+3:.1f}" fill="#8b949e" font-size="9" '
                         f'text-anchor="end">{lbl}</text>')
    parts.append(f'<text x="24" y="{(ty0+ty1)/2:.1f}" fill="#8b949e" font-size="10" '
                 f'transform="rotate(-90 24 {(ty0+ty1)/2:.1f})" text-anchor="middle">resolution (arcsec)</text>')
    poly = " ".join(f"{X(Ds[i]):.1f},{Y(arc[i]):.1f}" for i in range(len(Ds)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#4dabf7" stroke-width="2.6"/>')
    # mark the optical instruments (skip the radio one, different lambda)
    for name, D, l in insts:
        if l > 1e-6:
            continue
        a = rayleigh_angle_arcsec(l, D)
        parts.append(f'<circle cx="{X(D):.1f}" cy="{Y(a):.1f}" r="4" fill="#ffd43b"/>')
        parts.append(f'<text x="{X(D)+6:.1f}" y="{Y(a)+4:.1f}" fill="#ffd43b" font-size="9">'
                     f'{name.split(" (")[0]}</text>')

    # --- two Airy disks at the Rayleigh limit ---
    ay = size * 0.80
    sep = 70
    cx = size / 2
    parts.append(f'<text x="{x0:.1f}" y="{mid+28:.1f}" fill="#8b949e" font-size="12">'
                 f'two sources at the Rayleigh limit -- central peak of one over the first null of the other:</text>')
    for cxo, col in ((cx - sep / 2, "#06d6a0"), (cx + sep / 2, "#ff922b")):
        # Airy-ish: bright core + faint ring
        parts.append(f'<circle cx="{cxo:.1f}" cy="{ay:.1f}" r="30" fill="none" '
                     f'stroke="{col}" stroke-width="1" opacity="0.4"/>')
        parts.append(f'<circle cx="{cxo:.1f}" cy="{ay:.1f}" r="14" fill="{col}" opacity="0.85"/>')
    # summed intensity curve above
    n = 200
    ipts = []
    for k in range(n + 1):
        x = cx - 120 + 240 * k / n
        def airy(x0c):
            u = (x - x0c) / 22.0
            if abs(u) < 1e-6:
                return 1.0
            return (math.sin(u) / u) ** 2
        I = airy(cx - sep / 2) + airy(cx + sep / 2)
        ipts.append(f"{x:.1f},{ay - 60 - I * 34:.1f}")
    parts.append(f'<polyline points="{" ".join(ipts)}" fill="none" stroke="#e6edf3" stroke-width="1.8"/>')
    parts.append(f'<text x="{cx:.1f}" y="{ay+56:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">just resolved (the dip between peaks is the Rayleigh criterion)</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
