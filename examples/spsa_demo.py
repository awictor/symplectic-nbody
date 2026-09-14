"""SPSA demo: descend a 2-D bowl using two measurements per step, and show the eval-count win."""

import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import spsa


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


def _fmt(v):
    return f"{v:+.4f}"


def main():
    lines = []
    lines.append("SPSA -- Simultaneous Perturbation Stochastic Approximation")
    lines.append("=" * 62)
    lines.append("")
    lines.append("Gradient-free descent: 2 function evaluations per step, ANY dimension.")
    lines.append("")

    # --- 2-D bowl trajectory (also used for the SVG) ---
    center = [1.2, -0.8]
    f = spsa.shifted_sphere(center)
    theta0 = [-1.8, 1.9]
    theta, hist, evals = spsa.minimize(f, theta0, iterations=400, a=0.35, c=0.12,
                                       seed=1, track=True)
    lines.append(f"2-D bowl min at ({center[0]}, {center[1]}), start ({theta0[0]}, {theta0[1]})")
    lines.append(f"  after 400 iters ({evals} evals): "
                 f"({_fmt(theta[0])}, {_fmt(theta[1])})  f={f(theta):.2e}")
    lines.append("")

    # --- convergence table ---
    lines.append("  iter        f(theta)")
    lines.append("  " + "-" * 26)
    for k in (0, 25, 50, 100, 200, 399):
        lines.append(f"  {k:4d}    {hist[k][1]:12.6e}")
    lines.append("")

    # --- eval-count comparison across dimensions ---
    lines.append("Evals to run 100 optimization steps:")
    lines.append("   dim   finite-diff   SPSA   speedup")
    lines.append("   " + "-" * 36)
    for p in (2, 10, 50, 200):
        fd = 2 * p * 100
        sp = 2 * 100
        lines.append(f"  {p:4d}   {fd:10d}   {sp:5d}   {fd // sp:4d}x")
    lines.append("")

    # --- noisy objective ---
    g = spsa.noisy(spsa.sphere, noise_amp=0.05, seed=11)
    out = spsa.minimize(g, [2.0, -2.0, 1.5], iterations=3000, a=0.25, c=0.15, seed=5)
    dist = sum(x * x for x in out) ** 0.5
    lines.append(f"Noisy sphere (amp 0.05), 3-D: final |theta| = {dist:.4f}")
    lines.append("  -> converges despite noise that would thrash finite differences.")

    text = "\n".join(lines)
    print(text)

    svg = _svg(theta0, hist, center)
    return text, svg


def _svg(theta0, hist, center):
    W, H = 640, 420
    P = PALETTE
    # world extent covering the trajectory + optimum
    xs = [theta0[0], center[0]] + [h[0][0] for h in hist]
    ys = [theta0[1], center[1]] + [h[0][1] for h in hist]
    xmin, xmax = min(xs) - 0.4, max(xs) + 0.4
    ymin, ymax = min(ys) - 0.4, max(ys) + 0.4

    def sx(x):
        return 60 + (x - xmin) / (xmax - xmin) * (W - 90)

    def sy(y):
        return H - 60 - (y - ymin) / (ymax - ymin) * (H - 100)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="28" fill="{P["text"]}" font-size="15">'
                 f'SPSA descent on a 2-D bowl (2 evals/step)</text>')

    # concentric contour rings around the optimum (grayscale)
    cx, cy = sx(center[0]), sy(center[1])
    for i, r in enumerate((0.4, 0.9, 1.5, 2.2, 3.0)):
        rx = r / (xmax - xmin) * (W - 90)
        ry = r / (ymax - ymin) * (H - 100)
        shade = 40 + i * 10
        parts.append(f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" '
                     f'fill="none" stroke="#{shade:02x}{shade:02x}{shade:02x}" stroke-width="1"/>')

    # trajectory polyline
    pts = " ".join(f"{sx(h[0][0]):.1f},{sy(h[0][1]):.1f}" for h in hist)
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{P["yellow"]}" '
                 f'stroke-width="1.6" opacity="0.9"/>')

    # start (red) and optimum (green ring)
    parts.append(f'<circle cx="{sx(theta0[0]):.1f}" cy="{sy(theta0[1]):.1f}" r="6" '
                 f'fill="{P["red"]}"/>')
    parts.append(f'<text x="{sx(theta0[0]) + 10:.1f}" y="{sy(theta0[1]) + 4:.1f}" '
                 f'fill="{P["red"]}" font-size="12">start</text>')
    parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="8" fill="none" '
                 f'stroke="{P["green"]}" stroke-width="2.5"/>')
    parts.append(f'<text x="{cx + 12:.1f}" y="{cy + 4:.1f}" fill="{P["green"]}" '
                 f'font-size="12">optimum</text>')

    # final point (blue)
    fx, fy = sx(hist[-1][0][0]), sy(hist[-1][0][1])
    parts.append(f'<circle cx="{fx:.1f}" cy="{fy:.1f}" r="5" fill="{P["blue"]}"/>')

    parts.append(f'<text x="20" y="{H - 20}" fill="{P["gray"]}" font-size="11">'
                 f'yellow = mean trajectory over 400 steps; jitter = simultaneous perturbation</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
