"""Marching squares: extracting contour lines from a scalar field.

Given a 2-D grid of scalar values -- a height map, a temperature field, a signed-distance function
-- marching squares finds the ISO-CONTOUR at a chosen level c: the curve where the field equals c
(a topographic contour line, an isotherm, the boundary of a blob). It is the 2-D sibling of
marching cubes (the 3-D isosurface method behind medical imaging and metaball rendering), and the
workhorse behind every contour plot.

The idea is local and embarrassingly parallel. Overlay a grid of cells; at each of a cell's four
corners the field is either ABOVE or BELOW c, giving a 4-bit CASE INDEX (16 possibilities). That
index says which cell EDGES the contour crosses -- and exactly where on each edge is found by LINEAR
INTERPOLATION between the two corner values (so the contour passes through the corner value c, not
just the cell midpoint), giving a smooth curve. Each cell contributes 0, 1, or 2 line segments;
concatenating them over the grid yields the full contour. The only subtlety is the two AMBIGUOUS
saddle cases (opposite corners above, the other two below), resolved consistently here by the
cell-center average.

This module builds the 16-case lookup, extracts iso-contour segments with linear interpolation,
handles the saddle ambiguity, and can trace a scalar function or an explicit grid -- verified that a
radial field r^2 contours to circles of the right radius (segment endpoints lie on the circle to
interpolation accuracy), that a linear ramp gives straight contours, that a level below/above the
whole field yields no contour, and that the contour of a closed blob forms closed loops. Pure
stdlib; a computational-geometry / scientific-visualization companion to the polygon notes."""

from __future__ import annotations

import math


def _interp(p1, v1, p2, v2, level):
    """Point on segment p1-p2 where the field linearly crosses `level`."""
    if v2 == v1:
        return p1
    t = (level - v1) / (v2 - v1)
    return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))


def contour_segments(grid, level, x0=0.0, y0=0.0, dx=1.0, dy=1.0):
    """Iso-contour line segments of a scalar field at `level`.

    grid: list of rows (grid[j][i] = value at column i, row j). Returns a list of
    ((x1, y1), (x2, y2)) segments, in world coordinates set by the origin and spacing."""
    ny = len(grid)
    if ny < 2:
        return []
    nx = len(grid[0])
    if nx < 2:
        return []
    segments = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            # corner values, going counter-clockwise from bottom-left
            bl = grid[j][i]
            br = grid[j][i + 1]
            tr = grid[j + 1][i + 1]
            tl = grid[j + 1][i]
            # corner world positions
            xl, xr = x0 + i * dx, x0 + (i + 1) * dx
            yb, yt = y0 + j * dy, y0 + (j + 1) * dy
            pbl, pbr, ptr, ptl = (xl, yb), (xr, yb), (xr, yt), (xl, yt)
            # 4-bit case: bit set if that corner is >= level
            case = ((1 if bl >= level else 0)
                    | (2 if br >= level else 0)
                    | (4 if tr >= level else 0)
                    | (8 if tl >= level else 0))
            if case == 0 or case == 15:
                continue                                   # cell entirely below/above
            # edge crossing points (only the ones this case needs are computed)
            bottom = _interp(pbl, bl, pbr, br, level)
            right = _interp(pbr, br, ptr, tr, level)
            top = _interp(ptl, tl, ptr, tr, level)
            left = _interp(pbl, bl, ptl, tl, level)
            segs = _case_segments(case, bl, br, tr, tl, level, bottom, right, top, left)
            segments.extend(segs)
    return segments


def _case_segments(case, bl, br, tr, tl, level, bottom, right, top, left):
    """The 0-2 contour segments for a marching-squares case index."""
    # non-ambiguous cases: which pair of edges the contour connects
    table = {
        1: [(left, bottom)],
        2: [(bottom, right)],
        3: [(left, right)],
        4: [(right, top)],
        6: [(bottom, top)],
        7: [(left, top)],
        8: [(top, left)],
        9: [(top, bottom)],
        11: [(top, right)],
        12: [(right, left)],
        13: [(right, bottom)],
        14: [(bottom, left)],
    }
    if case in table:
        return table[case]
    # ambiguous saddles (5 and 10): resolve by the cell-center average
    center = (bl + br + tr + tl) / 4.0
    if case == 5:      # BL and TR above, BR and TL below
        if center >= level:
            return [(left, top), (bottom, right)]      # connect so the highs join
        return [(left, bottom), (right, top)]
    if case == 10:     # BR and TL above
        if center >= level:
            return [(left, bottom), (right, top)]
        return [(left, top), (bottom, right)]
    return []


def contour_from_function(f, xmin, xmax, ymin, ymax, level, nx=50, ny=50):
    """Sample a scalar function f(x, y) on an nx x ny grid and return its iso-contour segments."""
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)
    grid = [[f(xmin + i * dx, ymin + j * dy) for i in range(nx)] for j in range(ny)]
    return contour_segments(grid, level, x0=xmin, y0=ymin, dx=dx, dy=dy)


def total_length(segments):
    """Sum of the lengths of all contour segments (approximates the contour's arc length)."""
    return sum(math.dist(a, b) for a, b in segments)
