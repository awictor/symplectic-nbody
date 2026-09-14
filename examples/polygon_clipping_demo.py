"""Polygon clipping demo: clip a star polygon against a rectangle window, before/after (SVG)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import polygon_clipping as PC


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
BLUE = "#4dabf7"
GREEN = "#06d6a0"
YELLOW = "#ffd43b"
RED = "#ff6b6b"


def main(outdir=None):
    # a tilted convex pentagon straddling the window's corner (genuine partial overlap)
    subject = [(1, 5), (5, 1.5), (9.5, 4), (7, 9), (2, 8.5)]
    win = (4.0, 3.0, 8.0, 7.0)                    # xmin, ymin, xmax, ymax
    clipped = PC.clip_rectangle(subject, *win)

    lines = []
    lines.append("Sutherland-Hodgman polygon clipping")
    lines.append("=" * 50)
    lines.append(f"subject polygon: {len(subject)} vertices, area {PC.area(subject):.3f}")
    lines.append(f"clip window (rectangle): x in [{win[0]},{win[2]}], y in [{win[1]},{win[3]}], "
                 f"area {(win[2]-win[0])*(win[3]-win[1]):.1f}")
    lines.append("")
    lines.append(f"clipped polygon: {len(clipped)} vertices, area {PC.area(clipped):.3f}")
    lines.append("")
    lines.append("clipped vertices (all inside the window):")
    for v in clipped:
        inside = PC.point_in_convex(v, [(win[0], win[1]), (win[2], win[1]),
                                        (win[2], win[3]), (win[0], win[3])])
        lines.append(f"  ({v[0]:+.3f}, {v[1]:+.3f})   inside: {inside}")
    lines.append("")
    lines.append("Each rectangle edge whittles the polygon down to its inside half-plane;")
    lines.append("four edges later, what remains is exactly the polygon-window intersection.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        W, H = 640, 420
        ml, mt, sc = 60, 60, 34
        oy = 20

        def sx(x):
            return ml + x * sc

        def sy(y):
            return mt + (12 - y) * sc

        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s.append(f'<text x="{ml}" y="30" fill="{TEXT}" font-size="15">'
                 f'Sutherland-Hodgman: subject clipped to the rectangle window</text>')
        # window
        s.append(f'<rect x="{sx(win[0]):.1f}" y="{sy(win[3]):.1f}" '
                 f'width="{(win[2]-win[0])*sc:.1f}" height="{(win[3]-win[1])*sc:.1f}" '
                 f'fill="none" stroke="{YELLOW}" stroke-width="1.5" stroke-dasharray="5,3"/>')
        # subject polygon (blue outline)
        subj_pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in subject)
        s.append(f'<polygon points="{subj_pts}" fill="{BLUE}" fill-opacity="0.10" '
                 f'stroke="{BLUE}" stroke-width="1.5"/>')
        # clipped polygon (green filled)
        if clipped:
            clip_pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in clipped)
            s.append(f'<polygon points="{clip_pts}" fill="{GREEN}" fill-opacity="0.35" '
                     f'stroke="{GREEN}" stroke-width="2.5"/>')
            for x, y in clipped:
                s.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.5" fill="{GREEN}"/>')
        # subject vertices
        for x, y in subject:
            s.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="{BLUE}"/>')
        # legend
        s.append(f'<text x="{ml}" y="{H-40}" fill="{BLUE}" font-size="11">blue = subject polygon</text>')
        s.append(f'<text x="{ml}" y="{H-26}" fill="{YELLOW}" font-size="11">yellow dashed = clip window</text>')
        s.append(f'<text x="{ml}" y="{H-12}" fill="{GREEN}" font-size="11">'
                 f'green = clipped intersection (area {PC.area(clipped):.2f})</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "polygon_clipping.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
