"""Tests for MFCC: mel<->hz inverse, triangular filterbank, tone detection, DCT stage, discrimination."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mfcc import (  # noqa: E402
    hz_to_mel,
    mel_to_hz,
    mel_filterbank,
    hamming_window,
    frame_signal,
    power_spectrum,
    brute_power_spectrum,
    mfcc_frame,
    mfcc,
)
from dct import dct  # noqa: E402


PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def _tone(freq, sample_rate, n):
    return [math.sin(2 * math.pi * freq * t / sample_rate) for t in range(n)]


def main():
    # ---- 1. mel <-> hz round-trip and monotonicity --------------------------------------
    ok = all(abs(mel_to_hz(hz_to_mel(f)) - f) < 1e-6 for f in [0, 100, 440, 1000, 4000, 8000])
    check("mel<->hz round-trips", ok)
    check("hz_to_mel is monotincreasing", all(hz_to_mel(a) < hz_to_mel(b)
          for a, b in [(0, 100), (100, 1000), (1000, 8000)]))
    check("mel(0) = 0", abs(hz_to_mel(0.0)) < 1e-12)

    # ---- 2. mel filterbank is non-negative and triangular -------------------------------
    sr = 16000
    n_fft = 512
    fb = mel_filterbank(26, n_fft, sr)
    check("filterbank has 26 filters", len(fb) == 26)
    check("all filter weights non-negative", all(w >= 0 for f in fb for w in f))
    check("filters peak at 1.0", all(abs(max(f) - 1.0) < 1e-9 for f in fb), )
    # each filter is unimodal (rises then falls)
    unimodal = True
    for f in fb:
        peak = f.index(max(f))
        if any(f[k] > f[k + 1] + 1e-12 for k in range(peak)) or \
           any(f[k] < f[k + 1] - 1e-12 for k in range(peak, len(f) - 1)):
            unimodal = False
            break
    check("each filter is unimodal (triangular)", unimodal)

    # ---- 3. adjacent filters overlap and cover the band (sum > 0 across mid bins) --------
    total = [sum(f[k] for f in fb) for k in range(len(fb[0]))]
    mid = total[10:200]  # skip the very lowest/highest bins
    check("filters cover the mid band (overlap)", all(t > 0 for t in mid),
          f"min {min(mid):.3f}")

    # ---- 4. FFT power spectrum matches a brute-force DFT --------------------------------
    frame = _tone(1000, sr, 400)
    ps, nfft = power_spectrum(frame)
    bps = brute_power_spectrum(frame)
    max_err = max(abs(ps[k] - bps[k]) for k in range(len(ps)))
    check("FFT power spectrum == brute DFT", max_err < 1e-6, f"max err {max_err:.2e}")

    # ---- 5. a pure tone lights up the mel band containing its frequency -----------------
    freq = 2000
    frame = _tone(freq, sr, 512)
    win = hamming_window(len(frame))
    wframe = [frame[i] * win[i] for i in range(len(frame))]
    ps, nfft = power_spectrum(wframe)
    fb2 = mel_filterbank(26, nfft, sr)
    band_energy = [sum(f[k] * ps[k] for k in range(len(ps))) for f in fb2]
    top = band_energy.index(max(band_energy))
    # which mel bands cover freq's FFT bin?
    tone_bin = int(round(freq / sr * nfft))
    covering = [m for m in range(26) if fb2[m][tone_bin] > 0]
    check("tone energizes a band covering its frequency", top in covering or abs(top - covering[0]) <= 1,
          f"top band {top}, covering {covering}")

    # ---- 6. log-mel + DCT stage matches a direct DCT of the log energies ----------------
    fb3 = mel_filterbank(26, nfft, sr)
    mel_e = [math.log(sum(f[k] * ps[k] for k in range(len(ps))) + 1e-12) for f in fb3]
    coeffs_direct = dct(mel_e)[:13]
    coeffs = mfcc_frame(frame, sr, n_filters=26, n_coeffs=13)
    # mfcc_frame re-windows internally, so compute its own log energies for a fair match:
    win2 = hamming_window(len(frame))
    wf = [frame[i] * win2[i] for i in range(len(frame))]
    ps2, nfft2 = power_spectrum(wf)
    fb4 = mel_filterbank(26, nfft2, sr)
    mel_e2 = [math.log(sum(f[k] * ps2[k] for k in range(len(ps2))) + 1e-12) for f in fb4]
    expect = dct(mel_e2)[:13]
    match = max(abs(coeffs[i] - expect[i]) for i in range(13))
    check("mfcc_frame == manual mel->log->DCT", match < 1e-9, f"max diff {match:.2e}")

    # ---- 7. different tones give different MFCCs; same tone gives the same ---------------
    c_low = mfcc_frame(_tone(300, sr, 512), sr)
    c_high = mfcc_frame(_tone(3000, sr, 512), sr)
    c_low2 = mfcc_frame(_tone(300, sr, 512), sr)
    d_diff = sum((c_low[i] - c_high[i]) ** 2 for i in range(len(c_low))) ** 0.5
    d_same = sum((c_low[i] - c_low2[i]) ** 2 for i in range(len(c_low))) ** 0.5
    check("distinct tones -> distinct MFCC", d_diff > 1.0, f"dist {d_diff:.3f}")
    check("same tone -> identical MFCC", d_same < 1e-9, f"dist {d_same:.2e}")

    # ---- 8. coefficient 0 tracks overall energy ----------------------------------------
    quiet = [0.1 * math.sin(2 * math.pi * 500 * t / sr) for t in range(512)]
    loud = [1.0 * math.sin(2 * math.pi * 500 * t / sr) for t in range(512)]
    c_quiet = mfcc_frame(quiet, sr)
    c_loud = mfcc_frame(loud, sr)
    check("louder signal -> larger MFCC[0]", c_loud[0] > c_quiet[0], f"{c_loud[0]:.2f} vs {c_quiet[0]:.2f}")

    # ---- 9. framing covers the signal and pads the tail ---------------------------------
    sig = list(range(1000))
    frames = frame_signal(sig, 400, 160)
    check("frames are all frame_len long", all(len(f) == 400 for f in frames))
    check("first frame starts at 0", frames[0][0] == 0)

    # ---- 10. full mfcc returns one vector per frame -------------------------------------
    signal = _tone(440, sr, sr // 2)  # half a second
    feats = mfcc(signal, sr, n_coeffs=13)
    check("mfcc returns frames of 13 coeffs", len(feats) > 0 and all(len(row) == 13 for row in feats),
          f"{len(feats)} frames")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
