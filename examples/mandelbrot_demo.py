"""Demo: the Mandelbrot set -- infinite detail from z -> z^2 + c.

Prints escape times along the real axis and a small ASCII rendering, then draws the full set
coloured by escape time, revealing the cardioid, bulbs, and filamentary boundary.

    python examples/mandelbrot_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mandelbrot import (escape_time, in_set, in_main_cardioid,  # noqa: E402
                        in_period2_bulb, escaped_fraction)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Mandelbrot set: c stays bounded under z -> z^2 + c (escape when |z| > 2)\n")
    print("  Escape time along the real axis (in-set = 100):")
    for c_re in (-2.5, -2.0, -1.0, -0.5, 0.0, 0.25, 0.35, 0.5, 1.0):
        et = escape_time(complex(c_re, 0), 100)
        tag = "IN SET" if et >= 100 else f"escapes at {et}"
        print(f"    c = {c_re:>5.2f}  ->  {tag}")

    box = escaped_fraction(-2.0, 0.5, -1.25, 1.25, 60, 60)
    print("\n  The set fills %.0f%% of its [-2,0.5]x[-1.25,1.25] bounding box.\n" % ((1 - box) * 100))

    print("  ASCII view (# = in set):")
    _ascii()

    print("\n  One quadratic rule, iterated, produces a fractal of endless detail -- the")
    print("  cardioid body, the period-2 bulb at c=-1, ever-smaller bulbs around the edge, and")
    print("  a boundary that carries tiny copies of the whole set at every magnification.")

    _svg(os.path.join(outdir, "mandelbrot.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'mandelbrot.svg')}")


def _ascii():
    for j in range(22):
        im = 1.1 - 2.2 * j / 21
        row = []
        for i in range(56):
            re = -2.1 + 2.7 * i / 55
            row.append("#" if in_set(complex(re, im), 50) else " ")
        print("    " + "".join(row))


def _svg(path, size=720, pad=40):
    re_min, re_max = -2.1, 0.7
    im_min, im_max = -1.25, 1.25
    nx = size - 2 * pad
    ny = int(nx * (im_max - im_min) / (re_max - re_min))
    max_iter = 80

    def color(et):
        if et >= max_iter:
            return "#0d1117"                       # in set: background (black)
        t = et / max_iter
        # blue -> cyan -> yellow -> white ramp with escape time
        if t < 0.5:
            u = t / 0.5
            r = int(0x1b + u * (0x06 - 0x1b)); g = int(0x3a + u * (0xd6 - 0x3a)); b = int(0x8b + u * (0xf0 - 0x8b))
        else:
            u = (t - 0.5) / 0.5
            r = int(0x06 + u * (0xff - 0x06)); g = int(0xd6 + u * (0xd4 - 0xd6)); b = int(0xf0 + u * (0x3b - 0xf0))
        return f"#{max(0,min(255,r)):02x}{max(0,min(255,g)):02x}{max(0,min(255,b)):02x}"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">The Mandelbrot set</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'coloured by escape time: black = bounded (in the set), bright = fast escape near the edge</text>',
    ]

    top = 56
    px = (size - 2 * pad) / nx
    py = px
    # coarse grid to keep the SVG a reasonable size
    step = 3
    for i in range(0, nx, step):
        re = re_min + (re_max - re_min) * i / nx
        for j in range(0, ny, step):
            im = im_min + (im_max - im_min) * j / ny
            et = escape_time(complex(re, im), max_iter)
            col = color(et)
            if col == "#0d1117":
                continue
            parts.append(f'<rect x="{pad + i*px:.1f}" y="{top + j*py:.1f}" '
                         f'width="{px*step+0.6:.1f}" height="{py*step+0.6:.1f}" fill="{col}"/>')
    # draw the in-set region explicitly as the dark body so it reads as solid
    for i in range(0, nx, step):
        re = re_min + (re_max - re_min) * i / nx
        for j in range(0, ny, step):
            im = im_min + (im_max - im_min) * j / ny
            if in_set(complex(re, im), max_iter):
                parts.append(f'<rect x="{pad + i*px:.1f}" y="{top + j*py:.1f}" '
                             f'width="{px*step+0.6:.1f}" height="{py*step+0.6:.1f}" fill="#e6edf3"/>')

    parts.append(f'<text x="{pad:.1f}" y="{size-16:.1f}" fill="#8b949e" font-size="11">'
                 f'white = in the set; the boundary is an infinitely detailed fractal</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
