"""Demo: tanh-sinh quadrature -- integrating functions that blow up at the endpoints.

Integrates smooth and endpoint-singular functions, contrasting tanh-sinh with a naive Simpson rule
that chokes on the singularities. Draws the double-exponential clustering of abscissae toward the
endpoints.

    python examples/tanh_sinh_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tanh_sinh import integrate  # noqa: E402


def simpson(f, a, b, n=10000):
    if n % 2:
        n += 1
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * f(a + i * h)
    return s * h / 3


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Tanh-sinh (double-exponential) quadrature: taming endpoint singularities\n")

    cases = [
        ("x^2", lambda x: x * x, 0, 1, 1 / 3, False),
        ("exp(x)", math.exp, 0, 1, math.e - 1, False),
        ("1/sqrt(x)  (blows up at 0)", lambda x: 1 / math.sqrt(x) if x > 0 else 0.0, 0, 1, 2.0, True),
        ("ln(1/x)  (log blowup at 0)", lambda x: -math.log(x) if x > 0 else 0.0, 0, 1, 1.0, True),
        ("1/sqrt(1-x^2)  (blows up at 1)", lambda x: 1 / math.sqrt(1 - x * x) if x < 1 else 0.0,
         0, 1, math.pi / 2, True),
        ("1/sqrt(x(1-x))  (both ends)",
         lambda x: 1 / math.sqrt(x * (1 - x)) if 0 < x < 1 else 0.0, 0, 1, math.pi, True),
    ]

    print(f"  {'integrand':30s} {'exact':>12} {'tanh-sinh':>14} {'Simpson':>14}")
    for name, f, a, b, exact, singular in cases:
        ts = integrate(f, a, b)
        try:
            sp = simpson(f, a, b, 20000)
        except (ValueError, ZeroDivisionError, OverflowError):
            sp = float("nan")
        sp_str = f"{sp:.8f}" if math.isfinite(sp) else "diverges"
        print(f"  {name:30s} {exact:12.8f} {ts:14.8f} {sp_str:>14}")

    print("\n  On smooth integrands both methods nail it; on the singular ones tanh-sinh stays")
    print("  accurate to ~1e-8 while naive Simpson loses digits (its endpoint sample sits on or near")
    print("  the blow-up). The substitution x = tanh(pi/2 sinh t) clusters abscissae exponentially")
    print("  toward the endpoints without ever reaching them, and the transformed integrand decays")
    print("  double-exponentially, so a plain trapezoid rule in t converges with astonishing speed.")

    _svg(os.path.join(outdir, "tanh_sinh.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'tanh_sinh.svg')}")


def _svg(path, width=760, height=320):
    # show the abscissae u_k = tanh(pi/2 sinh(k h)) for a moderate h, on [-1,1]
    HALF_PI = math.pi / 2
    h = 0.25
    us = []
    k = 0
    while True:
        t = k * h
        u = math.tanh(HALF_PI * math.sinh(t))
        if 1 - abs(u) < 1e-9 and k > 0:
            us.append(u)
            break
        us.append(u)
        if k > 0:
            us.append(-u)
        k += 1
        if k > 40:
            break
    us = sorted(set(us))

    ox, oy = 40, 160
    span = width - 80

    def x_of(u):
        return ox + (u + 1) / 2 * span

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="17">'
        f'Tanh-sinh abscissae on [-1, 1]: they cluster toward the endpoints</text>',
        f'<text x="20" y="50" fill="#8b949e" font-size="12">'
        f'equally-spaced points in t map to exponentially-clustered points in x, taming singularities'
        f'</text>',
    ]

    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox+span}" y2="{oy}" stroke="#30363d"/>')
    for u in us:
        x = x_of(u)
        # density -> colour: nearer the ends = redder
        t = abs(u)
        col = f"rgb({int(80+175*t)},{int(180-120*t)},{int(220-150*t)})"
        parts.append(f'<line x1="{x:.1f}" y1="{oy-30}" x2="{x:.1f}" y2="{oy+30}" '
                     f'stroke="{col}" stroke-width="1.6"/>')
    parts.append(f'<text x="{x_of(-1):.0f}" y="{oy+55}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">-1</text>')
    parts.append(f'<text x="{x_of(1):.0f}" y="{oy+55}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">+1</text>')
    parts.append(f'<text x="{x_of(0):.0f}" y="{oy+55}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">0</text>')
    parts.append(f'<text x="{ox}" y="{oy+90}" fill="#8b949e" font-size="12">'
                 f'{len(us)} nodes shown; abscissae pile up near +-1 where singular integrands live'
                 f'</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
