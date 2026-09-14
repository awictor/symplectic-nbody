"""fBm demo: render fBm, turbulence, and ridged heightmaps -- terrain, clouds, mountains (SVG)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import fbm


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"


def _terrain_color(t):
    """Map height [0,1] to a terrain palette: water -> sand -> grass -> rock -> snow."""
    t = max(0.0, min(1.0, t))
    stops = [(0.0, (20, 40, 90)), (0.35, (40, 90, 170)), (0.42, (200, 190, 120)),
             (0.55, (60, 130, 60)), (0.75, (110, 90, 70)), (0.9, (150, 150, 150)),
             (1.0, (250, 250, 255))]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1:
            w = (t - t0) / (t1 - t0) if t1 > t0 else 0
            return f"#{int(c0[0]+w*(c1[0]-c0[0])):02x}{int(c0[1]+w*(c1[1]-c0[1])):02x}{int(c0[2]+w*(c1[2]-c0[2])):02x}"
    return "#fafaff"


def _gray(t):
    v = int(max(0.0, min(1.0, t)) * 255)
    return f"#{v:02x}{v:02x}{v:02x}"


def main(outdir=None):
    lines = []
    lines.append("Fractional Brownian motion: layered noise octaves")
    lines.append("=" * 52)
    lines.append("fBm(x) = sum gain^i * noise(lacunarity^i * x); each octave doubles the")
    lines.append("frequency and halves the amplitude -- self-similar detail across scales.")
    lines.append("")
    f = fbm.FBM(seed=7, octaves=6, lacunarity=2.0, gain=0.5)
    lines.append(f"octaves=6, lacunarity=2.0, gain=0.5")
    lines.append(f"max amplitude (geometric sum): {f.max_amplitude():.4f}")
    lines.append("")
    lines.append("total variation of a 1-D slice grows as octaves are added (more detail):")
    lines.append(f"{'octaves':>9}{'total variation':>18}")
    for oc in (1, 2, 3, 4, 6, 8):
        line = [fbm.FBM(seed=7, octaves=oc).fbm1(x * 0.05) for x in range(400)]
        lines.append(f"{oc:>9}{fbm.total_variation(line):>18.3f}")
    lines.append("")
    lines.append("variants: fBm = signed sum (terrain), turbulence = sum|noise| (clouds),")
    lines.append("ridged = sum(1-|noise|)^2 (sharp mountain ridges).")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        res = 120
        f = fbm.FBM(seed=7, octaves=6, lacunarity=2.0, gain=0.5)
        # panel 1: fBm as a colored terrain; panels 2,3: turbulence and ridged in grayscale
        px = 200
        W = 3 * px + 80
        H = px + 110
        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s.append(f'<text x="20" y="26" fill="{TEXT}" font-size="15">'
                 f'fBm terrain, turbulence (clouds), and ridged (mountains)</text>')
        cell = px / res
        panels = [("fBm terrain", "fbm", _terrain_color),
                  ("turbulence", "turbulence", _gray),
                  ("ridged", "ridged", _gray)]
        for pi, (title, mode, color) in enumerate(panels):
            ox = 20 + pi * (px + 20)
            oy = 50
            hm = f.heightmap(res, res, scale=0.03, mode=mode)
            s.append(f'<text x="{ox}" y="{oy-4}" fill="{GRAY}" font-size="11">{title}</text>')
            for j in range(res):
                for i in range(res):
                    s.append(f'<rect x="{ox+i*cell:.2f}" y="{oy+j*cell:.2f}" '
                             f'width="{cell+0.5:.2f}" height="{cell+0.5:.2f}" '
                             f'fill="{color(hm[j][i])}"/>')
        s.append(f'<text x="20" y="{H-14}" fill="{GRAY}" font-size="10">'
                 f'The same octave sum, reshaped: signed for rolling terrain, absolute for billowy '
                 f'clouds, inverted-and-squared for sharp ridges.</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "fbm.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
