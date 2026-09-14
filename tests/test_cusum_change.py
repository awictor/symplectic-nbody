"""Validate CUSUM: detects shifts, near change point, delay vs shift size, null run length, Page-Hinkley."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import cusum_change as cc


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


class _R:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF
        self._spare = None

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)

    def normal(self):
        if self._spare is not None:
            v = self._spare
            self._spare = None
            return v
        u1 = max(self.u(), 1e-12)
        u2 = self.u()
        r = math.sqrt(-2 * math.log(u1))
        self._spare = r * math.sin(2 * math.pi * u2)
        return r * math.cos(2 * math.pi * u2)


def main():
    print("CUSUM change-detection tests")

    # --- detects a mean shift and reports a change point near the truth ---
    rng = _R(1)
    change_at = 100
    stream = [rng.normal() for _ in range(change_at)] + [rng.normal() + 2.0 for _ in range(100)]
    res = cc.cusum(stream, target=0.0, sigma=1.0, k=0.5, h=5.0)
    check("shift detected", res["alarm_index"] is not None)
    check(f"alarm after the change ({res['alarm_index']} > {change_at})", res["alarm_index"] > change_at)
    check(f"alarm soon after change (delay {res['alarm_index'] - change_at})",
          res["alarm_index"] - change_at < 30)
    check("upward shift flagged as high", res["alarm_side"] == "high")

    # --- downward shift flagged as low ---
    stream_dn = [rng.normal() for _ in range(80)] + [rng.normal() - 2.0 for _ in range(80)]
    res_dn = cc.cusum(stream_dn, target=0.0, sigma=1.0, k=0.5, h=5.0)
    check("downward shift flagged as low", res_dn["alarm_side"] == "low")

    # --- stationary stream: rarely alarms (long run length) ---
    def gen(r):
        rr = _R(1000 + r)
        return [rr.normal() for _ in range(2000)]
    arl0 = cc.average_run_length(gen, n_runs=40, max_len=2000, k=0.5, h=5.0)
    check(f"stationary ARL is long ({arl0:.0f})", arl0 > 200)

    # --- larger shift detected sooner ---
    def delay_for_shift(shift, seed=7):
        rng = _R(seed)
        s = [rng.normal() for _ in range(100)] + [rng.normal() + shift for _ in range(200)]
        res = cc.cusum(s, 0.0, 1.0, k=0.5, h=5.0)
        return (res["alarm_index"] - 100) if res["alarm_index"] and res["alarm_index"] >= 100 else 999
    d1 = delay_for_shift(1.0)
    d3 = delay_for_shift(3.0)
    check(f"larger shift detected sooner ({d3} < {d1})", d3 < d1)

    # --- slack k suppresses drift under the null: sums return toward 0 ---
    rng = _R(3)
    stationary = [rng.normal() for _ in range(500)]
    res = cc.cusum(stationary, 0.0, 1.0, k=0.5, h=1e9)  # huge h so no alarm; inspect sums
    # the running sums should spend most time near 0, not grow unbounded
    mean_hi = sum(res["s_hi"]) / len(res["s_hi"])
    check(f"null CUSUM sums stay small (mean {mean_hi:.2f})", mean_hi < 2.0)
    # and hit zero repeatedly
    zeros = sum(1 for v in res["s_hi"] if v == 0.0)
    check("null CUSUM returns to zero often", zeros > 50)

    # --- bigger decision interval h lengthens run to false alarm ---
    arl_small = cc.average_run_length(gen, n_runs=30, max_len=3000, k=0.5, h=3.0)
    arl_big = cc.average_run_length(gen, n_runs=30, max_len=3000, k=0.5, h=6.0)
    check(f"bigger h -> longer ARL ({arl_small:.0f} < {arl_big:.0f})", arl_small < arl_big)

    # --- Page-Hinkley detects an upward shift without a preset target ---
    rng = _R(5)
    ph_stream = [rng.normal() for _ in range(100)] + [rng.normal() + 2.0 for _ in range(100)]
    ph = cc.page_hinkley(ph_stream, delta=0.5, threshold=8.0)
    check("Page-Hinkley detects shift", ph["alarm_index"] is not None)
    check(f"Page-Hinkley alarm after change ({ph['alarm_index']})", ph["alarm_index"] > 100)

    # --- Page-Hinkley: stationary rarely alarms ---
    rng = _R(9)
    stat_ph = cc.page_hinkley([rng.normal() for _ in range(500)], delta=0.5, threshold=8.0)
    check("Page-Hinkley stationary quiet", stat_ph["alarm_index"] is None)

    # --- no shift -> no alarm ---
    rng = _R(11)
    res = cc.cusum([rng.normal() for _ in range(300)], 0.0, 1.0, k=0.5, h=6.0)
    check("no shift -> no alarm (mostly)", res["alarm_index"] is None or res["alarm_index"] > 100)

    # --- deterministic ---
    a = cc.cusum(stream, 0.0, 1.0)
    b = cc.cusum(stream, 0.0, 1.0)
    check("deterministic", a["alarm_index"] == b["alarm_index"] and a["s_hi"] == b["s_hi"])

    # --- sums are non-negative (reflected at zero) ---
    check("CUSUM sums non-negative", all(v >= 0 for v in a["s_hi"]) and all(v >= 0 for v in a["s_lo"]))

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
