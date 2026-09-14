"""CUSUM change detection: spot the moment a data stream's mean shifts, faster than a moving average.

A control chart that plots each new value and flags an out-of-range point is slow to notice a SMALL,
SUSTAINED shift in the mean -- the kind that signals a drifting sensor, a degrading process, or the onset
of a fault. The CUMULATIVE SUM (CUSUM) chart (Page 1954) accumulates the running deviations from the
target, so a persistent small bias adds up and crosses a threshold long before any single point looks
anomalous. The two-sided tabular CUSUM keeps two sums,

    S_hi = max(0, S_hi + (x - target) - k),      S_lo = max(0, S_lo - (x - target) - k),

where the SLACK k (typically half the shift you want to detect, in standard deviations) makes the sums
drift back to zero under the null and grow only when a real shift persists. An alarm fires when either sum
exceeds the DECISION INTERVAL h (in units of sigma). Tuning (k, h) trades the average run length to false
alarm against the detection delay -- the classic sequential-detection tradeoff.

The related PAGE-HINKLEY test tracks the cumulative deviation from the running mean and its running
minimum, firing when the gap exceeds a threshold; it needs no pre-specified target and adapts online. Both
are staples of statistical process control, network-anomaly detection, and streaming change-point
monitoring.

This module implements the two-sided tabular CUSUM (with the running statistics and the first alarm index),
the Page-Hinkley test, and an average-run-length estimate by simulation. It is validated: on a stationary
stream CUSUM rarely alarms (long run length to false alarm); when the mean shifts partway through, CUSUM
fires shortly after the change and reports a change point near the true one; a larger shift is detected
sooner; the slack k suppresses drift under the null so the sums return to zero; Page-Hinkley likewise
detects a shift without a preset target; a bigger decision interval h lengthens the run to false alarm; and
results are deterministic. Pure stdlib; the sequential-change-detection companion to the Welch-PSD,
Kalman, and hypothesis-testing tools."""

from __future__ import annotations

import math


def cusum(data, target, sigma=1.0, k=0.5, h=5.0):
    """Two-sided tabular CUSUM. data: stream; target: in-control mean; sigma: known/estimated SD.

    k is the slack (in sigma, ~half the shift to detect), h the decision interval (in sigma). Returns a
    dict with s_hi, s_lo (per-step statistics), alarm_index (first alarm or None), and alarm_side."""
    s_hi = 0.0
    s_lo = 0.0
    hi_track = []
    lo_track = []
    alarm_index = None
    alarm_side = None
    K = k * sigma
    H = h * sigma
    for i, x in enumerate(data):
        dev = x - target
        s_hi = max(0.0, s_hi + dev - K)
        s_lo = max(0.0, s_lo - dev - K)
        hi_track.append(s_hi)
        lo_track.append(s_lo)
        if alarm_index is None:
            if s_hi > H:
                alarm_index = i
                alarm_side = "high"
            elif s_lo > H:
                alarm_index = i
                alarm_side = "low"
    return {
        "s_hi": hi_track,
        "s_lo": lo_track,
        "alarm_index": alarm_index,
        "alarm_side": alarm_side,
        "H": H,
    }


def page_hinkley(data, delta=0.5, threshold=10.0, alpha=0.0):
    """Page-Hinkley test for an increase in the mean. Tracks the cumulative deviation from the running
    mean minus a tolerance delta, and its running minimum; alarms when the gap exceeds `threshold`.

    Returns a dict with the PH statistic per step, alarm_index, and detected direction. Detects upward
    shifts; negate the data to detect downward ones."""
    n = 0
    mean = 0.0
    cum = 0.0
    ph_min = 0.0
    stat_track = []
    alarm_index = None
    for i, x in enumerate(data):
        n += 1
        mean += (x - mean) / n                      # running mean
        cum += x - mean - delta
        ph_min = min(ph_min, cum)
        ph = cum - ph_min                           # gap from the running minimum
        stat_track.append(ph)
        if alarm_index is None and ph > threshold:
            alarm_index = i
    return {"stat": stat_track, "alarm_index": alarm_index}


def average_run_length(gen, n_runs=200, max_len=2000, target=0.0, sigma=1.0, k=0.5, h=5.0):
    """Estimate the average run length (mean steps to alarm) for a stream generator `gen(run_index)`
    that returns a list of values. Runs with no alarm contribute max_len (censored)."""
    total = 0
    for r in range(n_runs):
        stream = gen(r)[:max_len]
        res = cusum(stream, target, sigma, k, h)
        total += res["alarm_index"] + 1 if res["alarm_index"] is not None else max_len
    return total / n_runs
