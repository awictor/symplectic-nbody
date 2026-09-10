"""Pure-Python SVG trajectory renderer. Zero dependencies.

Takes recorded body trajectories and writes a standalone .svg you can open in
any browser: orbit paths as polylines, start markers, end markers, and a small
energy-error readout. Projects the 3D positions onto a chosen 2D plane.
"""

from __future__ import annotations

from typing import List, Tuple

# A trajectory is, per body, a list of (x, y, z) samples.
Traj = List[List[Tuple[float, float, float]]]

_PALETTE = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261",
            "#8338ec", "#3a86ff", "#ff006e", "#06d6a0", "#ffbe0b"]


def _project(p, plane: str):
    x, y, z = p
    if plane == "xy":
        return x, y
    if plane == "xz":
        return x, z
    if plane == "yz":
        return y, z
    raise ValueError(f"unknown plane {plane!r}")


def render(traj: Traj, path: str, plane: str = "xy", size: int = 720,
           pad: int = 40, title: str = "", subtitle: str = "",
           background: str = "#0d1117") -> str:
    """Write an SVG of the trajectories to `path`. Returns the SVG string."""
    # world-space bounding box across every sample of every body
    xs, ys = [], []
    for body in traj:
        for p in body:
            u, v = _project(p, plane)
            xs.append(u); ys.append(v)
    if not xs:
        raise ValueError("no trajectory data")
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    spanx = (maxx - minx) or 1.0
    spany = (maxy - miny) or 1.0
    span = max(spanx, spany)  # uniform scale so orbits aren't distorted
    inner = size - 2 * pad

    def sx(u):
        return pad + (u - minx + (span - spanx) * 0.5) / span * inner

    def sy(v):
        # flip y so +y points up, like a physics plot
        return size - (pad + (v - miny + (span - spany) * 0.5) / span * inner)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="{background}"/>',
    ]

    for i, body in enumerate(traj):
        color = _PALETTE[i % len(_PALETTE)]
        pts = " ".join(f"{sx(u):.2f},{sy(v):.2f}"
                       for u, v in (_project(p, plane) for p in body))
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="1.4" stroke-opacity="0.9"/>'
        )
        # start (hollow) and end (filled) markers
        u0, v0 = _project(body[0], plane)
        u1, v1 = _project(body[-1], plane)
        parts.append(f'<circle cx="{sx(u0):.2f}" cy="{sy(v0):.2f}" r="3.5" '
                     f'fill="none" stroke="{color}" stroke-width="1.5"/>')
        parts.append(f'<circle cx="{sx(u1):.2f}" cy="{sy(v1):.2f}" r="4" fill="{color}"/>')

    if title:
        parts.append(f'<text x="{pad}" y="28" fill="#e6edf3" font-size="18">{_esc(title)}</text>')
    if subtitle:
        parts.append(f'<text x="{pad}" y="48" fill="#8b949e" font-size="12">{_esc(subtitle)}</text>')
    parts.append(f'<text x="{size - pad}" y="{size - 16}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">plane={plane}  hollow=start filled=end</text>')
    parts.append("</svg>")

    svg = "\n".join(parts)
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)
    return svg


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
