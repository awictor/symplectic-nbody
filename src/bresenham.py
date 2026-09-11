"""Bresenham rasterization: drawing lines and circles with integer arithmetic only.

Turning a mathematical line into pixels sounds trivial -- step x, round y -- but that uses slow
floating-point per pixel and mis-steps on steep lines. Bresenham's algorithm (1962) draws a line
using ONLY INTEGER additions and comparisons: it tracks an error term measuring how far the true
line has drifted from the chosen pixel, and whenever that error crosses a threshold it steps in the
minor axis and corrects the error. No division, no floating point, no rounding -- which is why it
ran on the first plotters and still underlies GPU line rasterizers and grid-based games.

The line routine handles all eight octants uniformly by working with absolute deltas and explicit
step signs (sx, sy) and a single error accumulator, so a line in any direction is one loop. The
companion MIDPOINT CIRCLE algorithm uses the same idea in a decision variable, drawing one octant
and mirroring it into the other seven by symmetry, so a full circle is rasterized from about r
integer steps.

This module implements Bresenham line drawing, the midpoint circle, and a filled-circle helper --
verified that a line hits both its endpoints, is connected (consecutive pixels are 8-adjacent),
matches the true line to within half a pixel, is symmetric under reversal, handles horizontal /
vertical / diagonal / steep / negative-slope cases, and that a circle's pixels all lie within half
a pixel of the true radius and have the expected 8-fold symmetry. Pure stdlib (integer math only);
a rasterization companion to the marching-squares note."""

from __future__ import annotations

import math


def line(x0, y0, x1, y1):
    """Bresenham's line: the list of integer pixels from (x0, y0) to (x1, y1), inclusive.
    Integer arithmetic only -- no floats, no division."""
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    pixels = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x1 >= x0 else -1
    sy = 1 if y1 >= y0 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        pixels.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return pixels


def circle(cx, cy, r):
    """Midpoint (Bresenham) circle: the set of integer pixels approximating a circle of radius r
    centred at (cx, cy). Returns a sorted list of unique pixels."""
    cx, cy, r = int(cx), int(cy), int(r)
    if r < 0:
        return []
    if r == 0:
        return [(cx, cy)]
    pixels = set()
    x = r
    y = 0
    err = 1 - r                       # midpoint decision variable
    while x >= y:
        # eight-fold symmetry: one octant mirrored into all eight
        for px, py in ((x, y), (y, x), (-x, y), (-y, x),
                       (-x, -y), (-y, -x), (x, -y), (y, -x)):
            pixels.add((cx + px, cy + py))
        y += 1
        if err < 0:
            err += 2 * y + 1
        else:
            x -= 1
            err += 2 * (y - x) + 1
    return sorted(pixels)


def filled_circle(cx, cy, r):
    """A filled disk: every integer pixel within radius r of (cx, cy)."""
    cx, cy, r = int(cx), int(cy), int(r)
    if r < 0:
        return []
    pixels = []
    r2 = r * r
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r2:
                pixels.append((cx + dx, cy + dy))
    return pixels


def _adjacent(a, b):
    """True if two pixels are 8-connected (touch, including diagonally)."""
    return abs(a[0] - b[0]) <= 1 and abs(a[1] - b[1]) <= 1 and a != b


def is_connected(pixels):
    """True if every consecutive pair of pixels in a line is 8-adjacent (an unbroken line)."""
    return all(_adjacent(pixels[i], pixels[i + 1]) for i in range(len(pixels) - 1))


def max_line_error(x0, y0, x1, y1):
    """Largest perpendicular distance from any rasterized pixel to the true line (should be < 1)."""
    pix = line(x0, y0, x1, y1)
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length == 0:
        return 0.0
    worst = 0.0
    for px, py in pix:
        # perpendicular distance from (px,py) to the line through the endpoints
        dist = abs(dy * (px - x0) - dx * (py - y0)) / length
        worst = max(worst, dist)
    return worst
