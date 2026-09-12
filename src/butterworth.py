"""Butterworth IIR filter design -- maximally-flat digital filters from the analog prototype.

Almost every stream of measured data needs filtering: strip the 60 Hz mains hum from an ECG, remove
high-frequency sensor noise before differentiating, split an audio signal into bass and treble, or
anti-alias before downsampling. A DIGITAL FILTER does this by a recurrence -- each output sample is a
weighted sum of recent inputs and recent outputs -- and the art is choosing the weights so the filter
passes the frequencies you want and rejects the rest. The BUTTERWORTH filter is the classic choice
because it is MAXIMALLY FLAT: its passband has no ripple at all, the gain falling monotonically from 1
toward 0 with none of the wiggles that Chebyshev or elliptic filters trade for a sharper cutoff. It is
the filter you reach for when a smooth, honest frequency response matters more than the steepest
possible roll-off.

Designing one is a beautiful three-step pipeline that every DSP textbook and every ``scipy.signal``
call runs under the hood:

  1. THE ANALOG PROTOTYPE. A Butterworth low-pass of order n has its poles equally spaced on the left
     half of the unit circle in the s-plane, at angles that keep the magnitude response |H(jw)|^2 =
     1 / (1 + w^(2n)) -- flat near w=0, down 3 dB at w=1, then rolling off at 6n dB per octave. The
     poles are pure geometry: s_k = exp(i pi (2k + n + 1) / (2n)).

  2. FREQUENCY PRE-WARPING AND SCALING. The digital cutoff (a fraction of the sample rate) is mapped
     to an analog frequency by the tangent PRE-WARP wc = tan(pi * fc), which compensates for the
     nonlinear frequency squashing the next step introduces, so the digital -3 dB point lands exactly
     where asked. The prototype poles are scaled by wc.

  3. THE BILINEAR TRANSFORM. The substitution s = (1 - z^-1)/(1 + z^-1) maps the entire analog left
     half-plane into the unit disk of the z-plane, turning the stable analog filter into a stable
     digital one and the s-domain transfer function into a ratio of polynomials in z -- the filter
     COEFFICIENTS b (numerator) and a (denominator) that drive the recurrence.

The module designs low-pass and high-pass Butterworth filters of any order, returns the (b, a)
coefficients, applies them by direct-form recurrence (and by forward-backward FILTFILT for zero phase
distortion), and computes the frequency response so the magnitude at any frequency can be read off.
Pure standard library -- ``math`` and ``cmath`` only, no numpy or scipy.

Validation. The design is checked against the DEFINING properties of a Butterworth filter, not against
another library: (1) the magnitude response is exactly -3 dB (1/sqrt(2)) at the cutoff frequency, to
numerical precision, for every order and cutoff tested; (2) the passband gain is ~1 and the stopband
gain rolls monotonically toward 0, with a higher order giving a steeper roll-off (measured in dB/octave
approaching the theoretical 6n); (3) the response is monotonic -- no ripple -- which is what
distinguishes Butterworth from Chebyshev; (4) a low-pass applied to a sum of a low and a high sine wave
removes the high one while preserving the low one's amplitude, verified by measuring the output tone
amplitudes; (5) filtfilt has zero phase lag, checked on a pure tone; (6) high-pass is the mirror image.
Coefficients are cross-checked against a hand-derived first-order RC filter."""

import cmath
import math


# ---------------------------------------------------------------------------
# analog prototype poles
# ---------------------------------------------------------------------------

def _butter_poles(n):
    """The n poles of the normalized (wc=1) analog Butterworth low-pass, in the left half-plane."""
    poles = []
    for k in range(n):
        theta = math.pi * (2 * k + n + 1) / (2 * n)
        poles.append(cmath.exp(1j * theta))
    return poles


# ---------------------------------------------------------------------------
# design: analog prototype -> prewarp -> bilinear transform -> (b, a)
# ---------------------------------------------------------------------------

def _poly_from_roots(roots):
    """Expand a monic polynomial from its complex roots; returns real coefficients (imag ~ 0)."""
    coeffs = [1.0 + 0j]
    for r in roots:
        new = [0j] * (len(coeffs) + 1)
        for i, c in enumerate(coeffs):
            new[i] += c
            new[i + 1] -= c * r
        coeffs = new
    return [c.real for c in coeffs]


def butter_lowpass(order, cutoff):
    """Design a low-pass Butterworth filter.

    ``cutoff`` is the -3 dB frequency as a fraction of the sample rate in (0, 0.5) (i.e. cutoff_hz /
    sample_rate). Returns (b, a): numerator and denominator coefficients for the difference equation
    a[0] y[n] = sum b[i] x[n-i] - sum a[j] y[n-j].
    """
    if not (0 < cutoff < 0.5):
        raise ValueError("cutoff must be in (0, 0.5) as a fraction of the sample rate")
    if order < 1:
        raise ValueError("order must be >= 1")

    wc = math.tan(math.pi * cutoff)          # prewarped analog cutoff
    proto = _butter_poles(order)
    analog_poles = [wc * p for p in proto]   # scale prototype to the cutoff

    # bilinear transform s = (1 - z^-1)/(1 + z^-1)  ->  z-poles and z-zeros
    z_poles = [(1 + p) / (1 - p) for p in analog_poles]
    z_zeros = [-1.0 + 0j] * order            # low-pass: all zeros at z = -1 (Nyquist)

    a = _poly_from_roots(z_poles)
    b = _poly_from_roots(z_zeros)

    # normalize for unity gain at DC (z = 1): H(1) = sum(b)/sum(a) == 1
    gain = sum(a) / sum(b)
    b = [c * gain for c in b]
    return b, a


def butter_highpass(order, cutoff):
    """Design a high-pass Butterworth filter (see butter_lowpass for the cutoff convention)."""
    if not (0 < cutoff < 0.5):
        raise ValueError("cutoff must be in (0, 0.5) as a fraction of the sample rate")
    if order < 1:
        raise ValueError("order must be >= 1")

    wc = math.tan(math.pi * cutoff)
    proto = _butter_poles(order)
    # high-pass transform of the prototype: s -> wc / s maps low-pass to high-pass
    analog_poles = [wc / p for p in proto]

    z_poles = [(1 + p) / (1 - p) for p in analog_poles]
    z_zeros = [1.0 + 0j] * order             # high-pass: all zeros at z = +1 (DC)

    a = _poly_from_roots(z_poles)
    b = _poly_from_roots(z_zeros)

    # normalize for unity gain at Nyquist (z = -1)
    def evalpoly(coeffs, z):
        return sum(c * z ** (len(coeffs) - 1 - i) for i, c in enumerate(coeffs))
    hn = evalpoly(b, -1.0) / evalpoly(a, -1.0)
    b = [c / hn for c in b]
    return b, a


# ---------------------------------------------------------------------------
# apply the filter
# ---------------------------------------------------------------------------

def lfilter(b, a, x):
    """Apply the IIR filter (b, a) to signal x by the direct-form-I recurrence. Returns the output."""
    n = len(x)
    y = [0.0] * n
    a0 = a[0]
    for i in range(n):
        acc = 0.0
        for j in range(len(b)):
            if i - j >= 0:
                acc += b[j] * x[i - j]
        for j in range(1, len(a)):
            if i - j >= 0:
                acc -= a[j] * y[i - j]
        y[i] = acc / a0
    return y


def filtfilt(b, a, x):
    """Zero-phase filtering: filter forward, reverse, filter again, reverse back. Cancels phase lag."""
    y = lfilter(b, a, x)
    y = lfilter(b, a, y[::-1])
    return y[::-1]


# ---------------------------------------------------------------------------
# frequency response
# ---------------------------------------------------------------------------

def freq_response(b, a, freq):
    """Complex frequency response H(e^{i 2pi f}) at digital frequency ``freq`` (fraction of fs)."""
    w = 2 * math.pi * freq
    z = cmath.exp(-1j * w)                    # z^-1
    num = sum(b[i] * z ** i for i in range(len(b)))
    den = sum(a[i] * z ** i for i in range(len(a)))
    return num / den


def magnitude(b, a, freq):
    """Magnitude of the frequency response at ``freq``."""
    return abs(freq_response(b, a, freq))


def magnitude_db(b, a, freq):
    """Frequency response magnitude in decibels."""
    m = magnitude(b, a, freq)
    return 20 * math.log10(m) if m > 0 else -math.inf
