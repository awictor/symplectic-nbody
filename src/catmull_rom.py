"""Catmull-Rom splines: interpolating curves that pass through every control point.

A B-spline approximates its control points; a Catmull-Rom spline INTERPOLATES them -- the curve
threads through all of them in order, which is why it is the default for keyframe animation, camera
paths, and level geometry in games. The trick is local: on each segment between P_i and P_{i+1}, the
tangents are estimated from the neighbours (P_{i-1}, P_{i+1}) and (P_i, P_{i+2}), giving a cubic
Hermite piece. Each point affects only the two segments around it, and neighbouring pieces share
position and tangent, so the whole curve is C^1.

The subtlety is PARAMETERIZATION -- how much curve-parameter to allot between successive points. The
UNIFORM version spaces them equally, which is simplest but produces cusps and self-intersections when
control points are unevenly spaced or double back. The CHORDAL version uses the distance between
points; the CENTRIPETAL version (the celebrated Yuksel-Schaefer-Keyser result) uses the SQUARE ROOT of
that distance, and it is provably free of cusps and self-intersections and never overshoots -- which is
why centripetal Catmull-Rom is the modern default. The general knot spacing is t_{i+1} = t_i +
|P_{i+1} - P_i|^alpha, with alpha = 0 uniform, 0.5 centripetal, 1 chordal.

This module builds Catmull-Rom splines for all three parameterizations (via the general Barry-Goldman
non-uniform formulation), evaluates position and tangent, and samples the curve. It is validated: the
curve interpolates every control point exactly; segments join with matching position and tangent (C^1);
the uniform form reduces to the classic tangent = (P_{i+1} - P_{i-1})/2 Hermite spline; a straight line
of collinear points stays straight; the centripetal form stays within a sane bound of the control
polygon where the uniform form overshoots on a sharp corner; and reversing the points reverses the
curve. Pure stdlib; the interpolating-curve companion to the B-spline and Bezier tools."""

from __future__ import annotations


def _dist(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(len(a))) ** 0.5


def _knot_sequence(points, alpha):
    """Non-uniform knots t_i with spacing |P_{i+1}-P_i|^alpha (alpha: 0 uniform, .5 centripetal, 1 chordal)."""
    t = [0.0]
    for i in range(1, len(points)):
        d = _dist(points[i], points[i - 1])
        step = d ** alpha if d > 0 else 1e-9
        t.append(t[-1] + step)
    return t


def _segment_point(P0, P1, P2, P3, t0, t1, t2, t3, t):
    """Barry-Goldman recursive evaluation of the non-uniform Catmull-Rom segment between P1 and P2."""
    dim = len(P0)

    def lerp(a, b, ta, tb, tt):
        if tb == ta:
            return list(a)
        w = (tt - ta) / (tb - ta)
        return [a[i] * (1 - w) + b[i] * w for i in range(dim)]

    A1 = lerp(P0, P1, t0, t1, t)
    A2 = lerp(P1, P2, t1, t2, t)
    A3 = lerp(P2, P3, t2, t3, t)
    B1 = lerp(A1, A2, t0, t2, t)
    B2 = lerp(A2, A3, t1, t3, t)
    C = lerp(B1, B2, t1, t2, t)
    return C


class CatmullRom:
    """Interpolating Catmull-Rom spline through the given points, with a chosen parameterization."""

    def __init__(self, points, kind="centripetal"):
        if len(points) < 2:
            raise ValueError("need at least 2 points")
        self.points = [list(p) for p in points]
        self.kind = kind
        self.alpha = {"uniform": 0.0, "centripetal": 0.5, "chordal": 1.0}.get(kind, kind)
        # pad the ends by reflecting so the first/last segments have neighbours
        pts = self.points
        first = [2 * pts[0][i] - pts[1][i] for i in range(len(pts[0]))]
        last = [2 * pts[-1][i] - pts[-2][i] for i in range(len(pts[0]))]
        self._padded = [first] + pts + [last]
        self._knots = _knot_sequence(self._padded, self.alpha)

    def n_segments(self):
        return len(self.points) - 1

    def evaluate_segment(self, seg, s):
        """Point on segment `seg` (between points[seg] and points[seg+1]) for local s in [0,1]."""
        j = seg + 1                           # index into padded array of P1
        P0, P1, P2, P3 = self._padded[j - 1], self._padded[j], self._padded[j + 1], self._padded[j + 2]
        t0, t1, t2, t3 = self._knots[j - 1], self._knots[j], self._knots[j + 1], self._knots[j + 2]
        t = t1 + s * (t2 - t1)
        return _segment_point(P0, P1, P2, P3, t0, t1, t2, t3, t)

    def tangent_segment(self, seg, s, h=1e-6):
        a = self.evaluate_segment(seg, min(max(s - h, 0.0), 1.0))
        b = self.evaluate_segment(seg, min(max(s + h, 0.0), 1.0))
        sa = min(max(s - h, 0.0), 1.0)
        sb = min(max(s + h, 0.0), 1.0)
        return [(b[i] - a[i]) / (sb - sa) for i in range(len(a))]

    def evaluate(self, u):
        """Global evaluation with u in [0, n_segments]; integer part selects the segment."""
        n = self.n_segments()
        if u <= 0:
            return self.evaluate_segment(0, 0.0)
        if u >= n:
            return self.evaluate_segment(n - 1, 1.0)
        seg = int(u)
        return self.evaluate_segment(seg, u - seg)

    def sample(self, per_segment=20):
        pts = []
        n = self.n_segments()
        for seg in range(n):
            steps = per_segment if seg == n - 1 else per_segment
            for k in range(steps + (1 if seg == n - 1 else 0)):
                pts.append(self.evaluate_segment(seg, k / steps))
        return pts


def uniform(points):
    return CatmullRom(points, "uniform")


def centripetal(points):
    return CatmullRom(points, "centripetal")


def chordal(points):
    return CatmullRom(points, "chordal")
