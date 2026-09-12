"""Demo: the unscented Kalman filter -- tracking a projectile seen only in range and bearing.

A sensor at the origin measures noisy range and bearing to a projectile flying under gravity. The UKF
recovers the full Cartesian position and velocity through the nonlinear measurement map, with no
Jacobians. Prints the tracking error and draws the true path, the noisy measurement-implied positions,
and the filtered estimate.

    python examples/unscented_kalman_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unscented_kalman import UnscentedKalmanFilter  # noqa: E402


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)

    def normal(self, sd=1.0):
        return sd * (sum(self.u() for _ in range(12)) - 6.0)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Unscented Kalman filter: Cartesian state from nonlinear range/bearing sensing\n")

    g = 9.8
    dt = 0.1

    def fx(s, dt):
        return [s[0] + dt * s[2], s[1] + dt * s[3], s[2], s[3] - g * dt]

    def hx(s):
        return [math.hypot(s[0], s[1]), math.atan2(s[1], s[0])]

    ukf = UnscentedKalmanFilter(4, 2, fx, hx)
    ukf.x = [2.0, 2.0, 28.0, 42.0]      # deliberately off from truth
    ukf.P = [[4.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    Q = [[0.01 if i == j else 0.0 for j in range(4)] for i in range(4)]
    r_sd, th_sd = math.sqrt(0.5), math.sqrt(0.001)
    R = [[r_sd ** 2, 0.0], [0.0, th_sd ** 2]]

    true = [0.0, 0.0, 30.0, 40.0]
    rng = LCG(2024)
    truth, meas_xy, est = [], [], []
    errs = []
    for step in range(80):
        true = fx(true, dt)
        if true[1] < 0:
            break
        r = math.hypot(true[0], true[1]) + rng.normal(r_sd)
        th = math.atan2(true[1], true[0]) + rng.normal(th_sd)
        ukf.predict(dt, Q)
        ukf.update([r, th], R)
        truth.append((true[0], true[1]))
        meas_xy.append((r * math.cos(th), r * math.sin(th)))
        est.append((ukf.x[0], ukf.x[1]))
        if step > 10:
            errs.append(math.hypot(ukf.x[0] - true[0], ukf.x[1] - true[1]))

    rms = math.sqrt(sum(e * e for e in errs) / len(errs))
    meas_rms = math.sqrt(sum(math.hypot(meas_xy[i][0] - truth[i][0],
                                        meas_xy[i][1] - truth[i][1]) ** 2
                             for i in range(11, len(truth))) / (len(truth) - 11))
    print(f"  tracked {len(truth)} steps of a projectile under gravity")
    print(f"    RMS position error -- raw measurements: {meas_rms:.3f} m")
    print(f"    RMS position error -- UKF estimate:     {rms:.3f} m   "
          f"({meas_rms/rms:.1f}x better)\n")
    print(f"    final true position   ({truth[-1][0]:.2f}, {truth[-1][1]:.2f})")
    print(f"    final UKF estimate    ({est[-1][0]:.2f}, {est[-1][1]:.2f})\n")

    print("  The measurement model (range, bearing) is nonlinear, so a linear Kalman filter cannot be")
    print("  applied directly. The UKF pushes a handful of sigma points through the true nonlinear map")
    print("  and recovers the mean and covariance from the transformed cloud -- accurate to second")
    print("  order, and needing no Jacobians at all.")

    _svg(os.path.join(outdir, "unscented_kalman.svg"), truth, meas_xy, est)
    print(f"\n  wrote {os.path.join(outdir, 'unscented_kalman.svg')}")


def _svg(path, truth, meas, est, width=760, height=430, pad=50):
    xs = [p[0] for p in truth] + [p[0] for p in meas] + [p[0] for p in est]
    ys = [p[1] for p in truth] + [p[1] for p in meas] + [p[1] for p in est]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    sx = (width - 2 * pad) / (xmax - xmin or 1)
    sy = (height - 2 * pad) / (ymax - ymin or 1)

    def px(x):
        return pad + (x - xmin) * sx

    def py(y):
        return height - pad - (y - ymin) * sy

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="17">'
        f'UKF projectile tracking from noisy range/bearing</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'orange dots: positions implied by raw noisy measurements; green: UKF; blue: true path</text>',
    ]
    # noisy measurement scatter
    for x, y in meas:
        parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="2.2" fill="#ff922b" opacity="0.6"/>')
    # true path
    tp = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in truth)
    parts.append(f'<polyline points="{tp}" fill="none" stroke="#4dabf7" stroke-width="2.5"/>')
    # UKF estimate
    ep = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in est)
    parts.append(f'<polyline points="{ep}" fill="none" stroke="#06d6a0" stroke-width="1.8" '
                 f'stroke-dasharray="5 3"/>')
    # sensor at origin
    parts.append(f'<circle cx="{px(0):.1f}" cy="{py(0):.1f}" r="5" fill="#ffd43b"/>')
    parts.append(f'<text x="{px(0)+8:.0f}" y="{py(0)+4:.0f}" fill="#ffd43b" font-size="11">sensor</text>')
    parts.append(f'<text x="{pad}" y="{height-14}" fill="#8b949e" font-size="11">'
                 f'blue true path, green UKF estimate -- the filter cuts through the measurement scatter</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
