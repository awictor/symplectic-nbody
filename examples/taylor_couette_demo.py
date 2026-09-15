"""Taylor-Couette demo: the exact v(r)=Ar+B/r profile, Rayleigh stability line, and Taylor-vortex onset."""

import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import taylor_couette as tc


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


def main():
    r1, r2 = 1.0, 2.0

    lines = []
    lines.append("Taylor-Couette flow -- viscous fluid between two rotating cylinders")
    lines.append("=" * 74)
    lines.append("")
    lines.append(f"Gap r1={r1} to r2={r2}. Steady circular Couette flow: v(r) = A r + B/r,")
    lines.append("the only motion is azimuthal, fixed by no-slip at each wall.")
    lines.append("")

    cases = [
        ("inner spins, outer still", 5.0, 0.0),
        ("solid-body (both equal)", 3.0, 3.0),
        ("outer spins faster", 1.0, 3.0),
    ]
    lines.append("   configuration              A         B        Rayleigh verdict")
    lines.append("   " + "-" * 62)
    for name, o1, o2 in cases:
        a, b = tc.couette_coeffs(r1, r2, o1, o2)
        stable = tc.is_rayleigh_stable(r1, r2, o1, o2)
        lines.append(f"   {name:26s} {a:6.3f}   {b:7.3f}    {'STABLE' if stable else 'UNSTABLE (centrifugal)'}")
    lines.append("")
    lines.append("Rayleigh's criterion: stable iff L^2 = (r^2 Omega)^2 does not fall outward.")
    lines.append("Inner-only rotation always fails it -> the flow is primed to overturn.")
    lines.append("")

    ratio = tc.rayleigh_marginal_ratio(r1, r2)
    lines.append(f"Marginal (potential-vortex) line: Omega2/Omega1 = (r1/r2)^2 = {ratio:.4f}")
    lines.append("  On this line A=0, angular momentum is constant, Phi=0 everywhere.")
    lines.append("")

    lines.append("Viscosity delays the overturn until the Taylor number crosses Ta_c ~ 1708:")
    lines.append("   Omega1    gap d    nu      Taylor number    vortices?")
    lines.append("   " + "-" * 52)
    for o1, rr2, nu in [(30.0, 1.1, 1.0), (60.0, 1.1, 1.0), (100.0, 1.1, 1.0), (2000.0, 1.1, 1.0)]:
        ta = tc.taylor_number(r1, rr2, o1, nu)
        unst = tc.is_taylor_unstable(r1, rr2, o1, nu)
        lines.append(f"   {o1:6.0f}    {rr2-r1:5.2f}   {nu:4.1f}    {ta:12.1f}     {'YES' if unst else 'no'}")
    lines.append("")
    lines.append("Below Ta_c viscosity damps the donut vortices; above it, Taylor's 1923 vortex stack appears.")

    text = "\n".join(lines)
    print(text)

    svg = _svg(r1, r2)
    return text, svg


def _svg(r1, r2):
    W, H = 640, 440
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="24" fill="{P["text"]}" font-size="15">'
                 f'Azimuthal velocity profile v(r) across the gap, three configurations</text>')

    # left plot: v(r) vs r
    x0, x1 = 70, 340
    y0, y1 = 60, 360
    vmax = 6.0
    def px(r): return x0 + (r - r1) / (r2 - r1) * (x1 - x0)
    def py(v): return y1 - v / vmax * (y1 - y0)

    # axes
    parts.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<text x="{(x0+x1)/2:.0f}" y="{y1+22}" fill="{P["gray"]}" font-size="10" text-anchor="middle">radius r</text>')
    parts.append(f'<text x="{x0-40}" y="{(y0+y1)/2:.0f}" fill="{P["gray"]}" font-size="10">v(r)</text>')
    parts.append(f'<text x="{x0}" y="{y1+22}" fill="{P["gray"]}" font-size="9" text-anchor="middle">r1</text>')
    parts.append(f'<text x="{x1}" y="{y1+22}" fill="{P["gray"]}" font-size="9" text-anchor="middle">r2</text>')

    profiles = [
        ("inner only", 5.0, 0.0, P["red"]),
        ("solid body", 3.0, 3.0, P["green"]),
        ("outer fast", 1.0, 3.0, P["blue"]),
    ]
    ly = 70
    for name, o1, o2, col in profiles:
        pts = []
        for i in range(41):
            r = r1 + (r2 - r1) * i / 40
            v = tc.velocity(r, r1, r2, o1, o2)
            pts.append(f"{px(r):.1f},{py(v):.1f}")
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="2"/>')
        parts.append(f'<text x="{x1-4}" y="{ly}" fill="{col}" font-size="10" text-anchor="end">{name}</text>')
        ly += 15

    # right: cylinder cross-section cartoon with Taylor vortices
    cx, cy = 490, 210
    R1, R2 = 40, 95
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{R2}" fill="none" stroke="{P["gray"]}" stroke-width="2"/>')
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{R1}" fill="{P["purple"]}" opacity="0.35" stroke="{P["purple"]}" stroke-width="2"/>')
    # rotation arrow on inner
    parts.append(f'<path d="M {cx+R1} {cy} A {R1} {R1} 0 0 1 {cx} {cy+R1}" fill="none" '
                 f'stroke="{P["yellow"]}" stroke-width="2" marker-end="url(#ah)"/>')
    parts.append(f'<defs><marker id="ah" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">'
                 f'<path d="M0,0 L8,4 L0,8 Z" fill="{P["yellow"]}"/></marker></defs>')
    # gap fill hint of vortices: small alternating circles in the annulus
    import math
    for k in range(8):
        ang = k / 8 * 2 * math.pi
        rr = (R1 + R2) / 2
        vx = cx + rr * math.cos(ang)
        vy = cy + rr * math.sin(ang)
        col = P["red"] if k % 2 == 0 else P["blue"]
        parts.append(f'<circle cx="{vx:.1f}" cy="{vy:.1f}" r="9" fill="none" stroke="{col}" stroke-width="1.5" opacity="0.8"/>')
    parts.append(f'<text x="{cx}" y="{cy+R2+22}" fill="{P["text"]}" font-size="10" text-anchor="middle">Taylor vortices (Ta &gt; Ta_c)</text>')

    parts.append(f'<text x="20" y="{H-26}" fill="{P["gray"]}" font-size="11">'
                 f'Left: exact v(r)=Ar+B/r for three drive configs. Right: above the critical Taylor number the</text>')
    parts.append(f'<text x="20" y="{H-10}" fill="{P["gray"]}" font-size="11">'
                 f'smooth flow buckles into a stack of counter-rotating donut vortices (red/blue rings).</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
