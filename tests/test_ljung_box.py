"""Validate Ljung-Box: white noise passes, AR(1) rejected, seasonal flagged, LB>BP, df adjustment."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import ljung_box as lb


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
    print("Ljung-Box tests")

    # --- white noise: fails to reject (large p) ---
    rng = _R(1)
    wn = [rng.normal() for _ in range(300)]
    res = lb.ljung_box(wn, lags=10)
    check(f"white noise: large p ({res['p_value']:.3f})", res["p_value"] > 0.1)
    check("white noise: is_white_noise True", lb.is_white_noise(wn, lags=10))
    check("df == lags (no model adjustment)", res["df"] == 10)

    # --- AR(1): strongly autocorrelated, rejected ---
    rng = _R(3)
    ar = [0.0]
    for _ in range(300):
        ar.append(0.8 * ar[-1] + rng.normal())
    ar = ar[1:]
    res = lb.ljung_box(ar, lags=10)
    check(f"AR(1): tiny p ({res['p_value']:.2e})", res["p_value"] < 1e-6)
    check("AR(1): large Q", res["Q"] > 100)
    check("AR(1): not white noise", not lb.is_white_noise(ar, lags=10))
    # lag-1 autocorrelation should be near 0.8
    check(f"AR(1): rho_1 near 0.8 ({res['autocorrelations'][0]:.2f})",
          abs(res["autocorrelations"][0] - 0.8) < 0.1)

    # --- seasonal signal flagged at the seasonal lag ---
    period = 12
    seasonal = [math.sin(2 * math.pi * t / period) + 0.2 * rng.normal() for t in range(240)]
    res = lb.ljung_box(seasonal, lags=period)
    check(f"seasonal: rejected ({res['p_value']:.2e})", res["p_value"] < 0.01)

    # --- Ljung-Box exceeds Box-Pierce (small-sample correction) ---
    q_lb = lb.ljung_box(ar, lags=10)["Q"]
    q_bp, _ = lb.box_pierce(ar, lags=10)
    check(f"Ljung-Box > Box-Pierce ({q_lb:.1f} > {q_bp:.1f})", q_lb > q_bp)

    # --- Q grows with more lags when autocorrelation is real ---
    q5 = lb.ljung_box(ar, lags=5)["Q"]
    q15 = lb.ljung_box(ar, lags=15)["Q"]
    check(f"Q grows with lags for AR(1) ({q5:.1f} < {q15:.1f})", q5 < q15)

    # --- df adjustment for fitted parameters lowers df (raises the bar) ---
    r_noadj = lb.ljung_box(ar, lags=10, model_df=0)
    r_adj = lb.ljung_box(ar, lags=10, model_df=2)
    check("model_df lowers df", r_adj["df"] == 8 and r_noadj["df"] == 10)
    # same Q, fewer df -> smaller p-value (chi2 sf larger x at smaller df is smaller)
    check("fewer df -> smaller p (same Q)", r_adj["p_value"] <= r_noadj["p_value"])

    # --- autocorrelations: rho_0 normalization, lag-0 == 1 excluded from output ---
    rho = lb.autocorrelations(wn, 5)
    check("rho_0 == 1", abs(rho[0] - 1.0) < 1e-12)
    check("|rho_k| <= 1", all(abs(r) <= 1.0 + 1e-9 for r in rho))

    # --- constant series: zero variance handled ---
    rho_const = lb.autocorrelations([5.0] * 20, 3)
    check("constant series autocorr handled", rho_const[0] == 1.0)

    # --- deterministic ---
    check("deterministic", lb.ljung_box(ar, lags=8) == lb.ljung_box(ar, lags=8))

    # --- MA-like short-memory: autocorrelation only at lag 1 ---
    rng = _R(5)
    e = [rng.normal() for _ in range(301)]
    ma = [e[t] + 0.7 * e[t - 1] for t in range(1, 301)]
    res = lb.ljung_box(ma, lags=10)
    check(f"MA(1): rejected ({res['p_value']:.2e})", res["p_value"] < 0.01)
    check(f"MA(1): rho_1 positive, rho_2 small ({res['autocorrelations'][0]:.2f}, {res['autocorrelations'][1]:.2f})",
          res["autocorrelations"][0] > 0.3 and abs(res["autocorrelations"][1]) < 0.2)

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
