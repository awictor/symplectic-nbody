"""Tests for the spectrogram (STFT): flat tone ridge, rising chirp, Parseval, windows, shapes."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import spectrogram as SP  # noqa: E402


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


def main():
    N = 512

    # ---- 1. a steady sinusoid gives a flat frequency ridge ------------------------------
    tone = [math.cos(2 * math.pi * 8 * n / 64) for n in range(N)]
    track = SP.dominant_frequency_track(tone, frame_len=64, hop=32, sample_rate=64)
    check("steady tone -> flat ridge at 8 Hz", all(abs(f - 8.0) < 1e-6 for f in track), f"{track[:4]}")

    # ---- 2. a linear chirp gives a monotonically rising ridge --------------------------
    fs = 128
    chirp = [math.cos(2 * math.pi * (2 + 18 * (n / N)) * n / fs) for n in range(N)]
    ctrack = SP.dominant_frequency_track(chirp, frame_len=64, hop=16, sample_rate=fs)
    check("chirp ridge rises monotonically",
          all(ctrack[i] <= ctrack[i + 1] + 2.0 for i in range(len(ctrack) - 1)))
    check("chirp ridge ends higher than it starts", ctrack[-1] > ctrack[0] + 10, f"{ctrack[0]}..{ctrack[-1]}")

    # ---- 3. frame and bin counts match the framing parameters ---------------------------
    spec = SP.spectrogram(tone, frame_len=64, hop=32)
    n_frames = (N + 32 - 1) // 32
    check("frame count matches framing", len(spec) == n_frames, f"{len(spec)} vs {n_frames}")
    check("bin count is frame_len//2 + 1", len(spec[0]) == 33)

    # ---- 4. Parseval per frame (rectangular window) ------------------------------------
    frames = SP.frame_signal(tone, 64, 64)
    spectra = SP.stft(tone, 64, 64, "rectangular")
    ok = all(abs(SP.frame_energy(frames[i]) - SP.spectral_energy(spectra[i])) < 1e-6
             for i in range(len(frames)))
    check("Parseval holds per frame (rectangular)", ok)

    # ---- 5. window functions have the expected shape -----------------------------------
    h = SP.hann_window(8)
    check("Hann window endpoints are 0", abs(h[0]) < 1e-12 and abs(h[-1]) < 1e-12)
    check("Hann window peaks in the middle", max(h) == h[len(h) // 2] or max(h) == h[len(h) // 2 - 1])
    check("Hann window is symmetric", all(abs(h[k] - h[len(h) - 1 - k]) < 1e-12 for k in range(len(h))))
    hm = SP.hamming_window(8)
    check("Hamming window endpoints are 0.08", abs(hm[0] - 0.08) < 1e-9)
    check("rectangular window is all ones", SP.rectangular_window(5) == [1.0] * 5)

    # ---- 6. a silent signal gives a zero spectrogram ------------------------------------
    silent = [0.0] * N
    spec = SP.spectrogram(silent, 64, 32)
    check("silent signal -> zero spectrogram", all(v == 0 for row in spec for v in row))

    # ---- 7. two-tone signal shows two ridges --------------------------------------------
    two = [math.cos(2 * math.pi * 5 * n / 64) + math.cos(2 * math.pi * 20 * n / 64) for n in range(N)]
    spec = SP.spectrogram(two, 64, 64, "hann")
    # in a middle frame, bins 5 and 20 should both be prominent
    mags = spec[2]
    top2 = sorted(range(1, len(mags)), key=lambda k: -mags[k])[:2]
    check("two-tone signal shows peaks at bins 5 and 20", set(top2) == {5, 20}, f"{sorted(top2)}")

    # ---- 8. frame_signal zero-pads the final frame --------------------------------------
    frames = SP.frame_signal([1, 2, 3, 4, 5], 4, 4)
    check("frame_signal zero-pads the tail", frames[-1] == [5, 0, 0, 0], f"{frames[-1]}")
    check("frame_signal frame length correct", all(len(f) == 4 for f in frames))

    # ---- 9. Hann windowing reduces spectral leakage vs rectangular ----------------------
    # an off-bin tone (7.5 cycles) leaks; Hann should concentrate energy near the peak more
    leaky = [math.cos(2 * math.pi * 7.5 * n / 64) for n in range(64)]
    rect_spec = [abs(v) for v in SP.stft(leaky, 64, 64, "rectangular")[0]]
    hann_spec = [abs(v) for v in SP.stft(leaky, 64, 64, "hann")[0]]
    # measure leakage as energy outside the two bins nearest 7.5 (bins 7,8)
    def leakage(spec):
        total = sum(spec[:33])
        near = spec[7] + spec[8]
        return (total - near) / total if total > 0 else 0
    check("Hann reduces spectral leakage vs rectangular", leakage(hann_spec) < leakage(rect_spec),
          f"rect {leakage(rect_spec):.3f} hann {leakage(hann_spec):.3f}")

    # ---- 10. dominant track works with a different sample rate --------------------------
    fs = 1000
    sig = [math.cos(2 * math.pi * 100 * n / fs) for n in range(N)]   # 100 Hz tone
    track = SP.dominant_frequency_track(sig, frame_len=100, hop=50, sample_rate=fs)
    check("100 Hz tone tracked at ~100 Hz", all(abs(f - 100) <= 10 for f in track), f"{track[:3]}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
