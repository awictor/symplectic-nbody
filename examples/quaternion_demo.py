"""Demo: quaternions -- rotation composition, and why slerp beats naive interpolation.

Composes rotations, converts between representations, and compares SLERP (constant angular velocity)
to component-wise LERP (which speeds up mid-arc). Draws the tip of a rotating vector interpolated both
ways, showing slerp's even spacing versus lerp's bunching.

    python examples/quaternion_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quaternion import (from_axis_angle, from_euler, to_euler, rotate_vector,  # noqa: E402
                        multiply, slerp, angle_between, normalize)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Quaternions: gimbal-lock-free rotation and smooth interpolation\n")

    # compose two rotations
    q1 = from_axis_angle((0, 0, 1), math.radians(90))    # yaw 90
    q2 = from_axis_angle((0, 1, 0), math.radians(90))    # pitch 90
    combined = multiply(q2, q1)
    v = (1, 0, 0)
    print("  composing rotations (apply yaw 90 then pitch 90 to the x-axis):")
    print(f"    after yaw 90:        {fmt(rotate_vector(q1, v))}")
    print(f"    after pitch 90 too:  {fmt(rotate_vector(combined, v))}\n")

    # representation round trip
    q = from_euler(math.radians(30), math.radians(45), math.radians(60))
    r, p, y = to_euler(q)
    print("  representation round trip (Euler 30/45/60 deg):")
    print(f"    -> quaternion {fmt4(q)}")
    print(f"    -> back to Euler {math.degrees(r):.1f}/{math.degrees(p):.1f}/{math.degrees(y):.1f} deg\n")

    # slerp vs lerp angular velocity
    qa = from_axis_angle((0, 0, 1), 0.0)
    qb = from_axis_angle((0, 0, 1), math.radians(170))
    print("  slerp vs component-lerp over a 170-degree turn (angle from start at each step):")
    print("    t      slerp    lerp")
    for i in range(0, 11, 2):
        t = i / 10
        s_ang = math.degrees(angle_between(qa, slerp(qa, qb, t)))
        lq = normalize(tuple(qa[j] + t * (qb[j] - qa[j]) for j in range(4)))
        l_ang = math.degrees(angle_between(qa, lq))
        print(f"    {t:.1f}   {s_ang:6.1f}   {l_ang:6.1f}")
    print("    (slerp increments evenly; lerp lags then rushes -- uneven angular velocity)\n")

    print("  A unit quaternion (cos(theta/2), sin(theta/2)*axis) encodes a rotation; composing them is")
    print("  one multiply, and slerp walks the great-circle arc at constant speed. Four numbers, one")
    print("  unit-length constraint, no gimbal lock -- why aerospace and graphics run on quaternions.")

    _svg(os.path.join(outdir, "quaternion.svg"), qa, qb)
    print(f"\n  wrote {os.path.join(outdir, 'quaternion.svg')}")


def fmt(v):
    return "(" + ", ".join(f"{c:+.3f}" for c in v) + ")"


def fmt4(q):
    return "(" + ", ".join(f"{c:+.3f}" for c in q) + ")"


def _svg(path, qa, qb, width=760, height=430):
    cx, cy, R = width // 2, height // 2 + 20, 150

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="17">'
        f'SLERP vs LERP: interpolating a rotating vector over 170 degrees</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'green (slerp) points are evenly spaced on the arc; orange (lerp) bunch at the ends</text>',
    ]
    # reference arc
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="none" stroke="#21262d" '
                 f'stroke-width="1"/>')

    def tip(q):
        v = rotate_vector(q, (1, 0, 0))
        return cx + v[0] * R, cy - v[1] * R

    # slerp points (upper) and lerp points (drawn with same rotation but marked separately)
    for i in range(21):
        t = i / 20
        s = slerp(qa, qb, t)
        lq = normalize(tuple(qa[j] + t * (qb[j] - qa[j]) for j in range(4)))
        sx, sy = tip(s)
        lx, ly = tip(lq)
        parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="4" fill="#06d6a0"/>')
        parts.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="2.6" fill="#ff922b" opacity="0.9"/>')

    # center + start/end rays
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="3" fill="#8b949e"/>')
    for q, lab, col in ((qa, "start", "#4dabf7"), (qb, "end", "#4dabf7")):
        tx, ty = tip(q)
        parts.append(f'<line x1="{cx}" y1="{cy}" x2="{tx:.1f}" y2="{ty:.1f}" stroke="{col}" '
                     f'stroke-width="1.5" stroke-dasharray="4 3"/>')
        parts.append(f'<text x="{tx:.0f}" y="{ty-8:.0f}" fill="{col}" font-size="11" '
                     f'text-anchor="middle">{lab}</text>')

    parts.append(f'<text x="{cx-90}" y="{height-16}" fill="#06d6a0" font-size="12">slerp: even</text>')
    parts.append(f'<text x="{cx+20}" y="{height-16}" fill="#ff922b" font-size="12">lerp: bunched</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
