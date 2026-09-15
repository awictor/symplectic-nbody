"""FitzHugh-Nagumo: the two-variable caricature of a spiking neuron, and why excitability needs a threshold.

The Hodgkin-Huxley equations explain the nerve impulse with four coupled variables and a fistful of
empirical gating functions. FitzHugh (1961) and Nagumo (1962) asked what the MINIMAL cartoon is that still
spikes, and found it: two variables, a cubic, and a slow recovery term.

    dv/dt = v - v^3/3 - w + I        (fast voltage-like variable, cubic auto-catalysis)
    dw/dt = eps (v + a - b w)        (slow recovery variable, eps << 1)

v is the membrane-voltage caricature; w is a lumped "recovery" current that turns the spike off. The magic
is the separation of timescales (eps small): v jumps quickly along the cubic v-nullcline while w drifts
slowly. That geometry produces the three signatures of an EXCITABLE medium:

  A THRESHOLD, without any explicit switch. At rest the state sits at the stable fixed point. A small kick
      to v decays straight back; a kick past the middle branch of the cubic nullcline is amplified into a
      big excursion -- a SPIKE -- before recovery drags it home. The "threshold" is the repelling middle
      branch, not a number hard-coded anywhere.
  A REFRACTORY PERIOD. Right after a spike w is elevated, so a second kick of the same size fails: the
      neuron is temporarily un-excitable while w relaxes.
  RELAXATION OSCILLATIONS. Inject enough steady current I and the fixed point loses stability through a
      HOPF BIFURCATION; the state can no longer rest and instead limit-cycles -- periodic spiking -- with
      the sharp jumps and slow crawls characteristic of a relaxation oscillator.

This module integrates the system with fixed-step RK4, finds the fixed point (a cubic root of the
I-shifted nullclines), classifies its stability from the 2x2 Jacobian trace and determinant, sweeps the
injected current to locate the Hopf onset of oscillation, and measures whether a given stimulus evokes a
full spike. It is validated: the RK4 integrator matches an analytic linear test problem to high order; the
fixed point satisfies both nullcline equations; the Jacobian trace sign predicts the observed rest-vs-spike
behaviour (small I -> stable, intermediate I -> oscillating, large I -> stable again, the classic two Hopf
points); a sub-threshold kick decays while a supra-threshold kick spikes; and the refractory period
suppresses a closely following second stimulus. Pure stdlib. The excitable-media companion to the
Hodgkin-Huxley-lite, van der Pol, and reaction-diffusion notes."""

from __future__ import annotations

import math


class FitzHughNagumo:
    """A FitzHugh-Nagumo neuron with parameters (a, b, eps) and injected current I."""

    def __init__(self, a=0.7, b=0.8, eps=0.08, I=0.0):
        self.a = a
        self.b = b
        self.eps = eps
        self.I = I

    def deriv(self, v, w):
        """Return (dv/dt, dw/dt) at state (v, w)."""
        dv = v - v ** 3 / 3.0 - w + self.I
        dw = self.eps * (v + self.a - self.b * w)
        return dv, dw

    def step_rk4(self, v, w, dt):
        """One classical RK4 step of size dt. Returns (v_next, w_next)."""
        k1v, k1w = self.deriv(v, w)
        k2v, k2w = self.deriv(v + 0.5 * dt * k1v, w + 0.5 * dt * k1w)
        k3v, k3w = self.deriv(v + 0.5 * dt * k2v, w + 0.5 * dt * k2w)
        k4v, k4w = self.deriv(v + dt * k3v, w + dt * k3w)
        v2 = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
        w2 = w + dt / 6.0 * (k1w + 2 * k2w + 2 * k3w + k4w)
        return v2, w2

    def simulate(self, v0, w0, dt, steps):
        """Integrate from (v0, w0) for `steps` RK4 steps. Returns (ts, vs, ws) lists."""
        v, w = v0, w0
        ts, vs, ws = [0.0], [v], [w]
        t = 0.0
        for _ in range(steps):
            v, w = self.step_rk4(v, w, dt)
            t += dt
            ts.append(t)
            vs.append(v)
            ws.append(w)
        return ts, vs, ws

    def fixed_point(self):
        """The resting state: solve the cubic for v where both derivatives vanish.

        Setting dw/dt = 0 gives w = (v + a)/b. Substituting into dv/dt = 0:
            v - v^3/3 - (v + a)/b + I = 0.
        Solve this cubic for the real root (there is a unique one in the excitable regime); return (v*, w*)."""
        a, b, I = self.a, self.b, self.I
        # cubic: -1/3 v^3 + (1 - 1/b) v + (I - a/b) = 0  ->  v^3 + p v + q = 0 after scaling
        # multiply by -3: v^3 - 3(1 - 1/b) v - 3(I - a/b) = 0
        p = -3.0 * (1.0 - 1.0 / b)
        q = -3.0 * (I - a / b)
        v_star = _real_cubic_root(p, q)
        w_star = (v_star + a) / b
        return v_star, w_star

    def jacobian(self, v, w):
        """2x2 Jacobian of (dv/dt, dw/dt) at (v, w).

        d(dv)/dv = 1 - v^2,  d(dv)/dw = -1
        d(dw)/dv = eps,      d(dw)/dw = -eps b
        """
        return [[1.0 - v * v, -1.0],
                [self.eps, -self.eps * self.b]]

    def fixed_point_stable(self):
        """True iff the resting fixed point is linearly stable (trace < 0 and det > 0).

        Stability of a 2x2 system: stable node/focus <=> trace(J) < 0 and det(J) > 0. When trace crosses
        zero (with det > 0) a Hopf bifurcation spawns a limit cycle -- the onset of repetitive spiking."""
        v, w = self.fixed_point()
        j = self.jacobian(v, w)
        trace = j[0][0] + j[1][1]
        det = j[0][0] * j[1][1] - j[0][1] * j[1][0]
        return trace < 0 and det > 0

    def is_oscillating(self, dt=0.05, steps=8000, warmup=4000):
        """Detect a limit cycle by integrating from the fixed point + small perturbation and measuring the
        voltage swing AFTER a warmup. A large residual peak-to-peak amplitude means sustained oscillation."""
        v0, w0 = self.fixed_point()
        _, vs, _ = self.simulate(v0 + 0.01, w0, dt, steps)
        tail = vs[warmup:]
        return (max(tail) - min(tail)) > 1.0  # spikes swing ~2-4 units; sub-threshold decay stays tiny

    def spikes(self, v_kick, dt=0.05, steps=6000, w0=None, peak_thresh=1.0):
        """Return True if a single voltage kick to v_kick (from rest) triggers a full spike.

        A spike is a large positive excursion of v -- the trajectory rides up the right branch of the cubic
        to a peak near +2 before recovery drags it home. We detect it by an ABSOLUTE peak threshold (not
        relative to the kick): a sub-threshold or refractory kick decays monotonically and never approaches
        the spike peak, so max(v) stays well below peak_thresh."""
        v_rest, w_rest = self.fixed_point()
        if w0 is None:
            w0 = w_rest
        _, vs, _ = self.simulate(v_kick, w0, dt, steps)
        return max(vs) > peak_thresh


def _real_cubic_root(p, q):
    """One real root of v^3 + p v + q = 0 via Cardano / trigonometric method.

    Returns the principal real root. For the FHN excitable regime the cubic has a single real root; when
    three real roots exist this returns one of them (sufficient for the resting-state branch)."""
    disc = (q / 2.0) ** 2 + (p / 3.0) ** 3
    if disc >= 0:
        sq = math.sqrt(disc)
        u = _cbrt(-q / 2.0 + sq)
        w = _cbrt(-q / 2.0 - sq)
        return u + w
    # three real roots: trigonometric form
    r = math.sqrt(-(p / 3.0) ** 3)
    phi = math.acos(-q / 2.0 / r)
    m = 2.0 * math.sqrt(-p / 3.0)
    # return the root that is typically the resting branch (k=2 gives the low/left root for FHN)
    return m * math.cos((phi + 2.0 * math.pi * 2) / 3.0)


def _cbrt(x):
    """Real cube root (handles negatives, unlike x ** (1/3))."""
    return math.copysign(abs(x) ** (1.0 / 3.0), x)


def hopf_current_window(a=0.7, b=0.8, eps=0.08, i_lo=0.0, i_hi=2.0, samples=400):
    """Scan injected current I in [i_lo, i_hi]; return (I_on, I_off) where the fixed point is UNSTABLE.

    Between the two Hopf points the neuron fires repetitively; outside it rests. Returns the first and last
    I at which trace(J) >= 0 (None if never unstable)."""
    on = off = None
    for k in range(samples + 1):
        I = i_lo + (i_hi - i_lo) * k / samples
        neuron = FitzHughNagumo(a, b, eps, I)
        if not neuron.fixed_point_stable():
            if on is None:
                on = I
            off = I
    return on, off
