"""Ljung-Box test: are these residuals white noise, or is there autocorrelation a model left behind?

After fitting a time-series model you must check the RESIDUALS: if they still carry autocorrelation, the
model missed structure and its forecasts and standard errors are wrong. Eyeballing the autocorrelation
function is subjective; the LJUNG-BOX TEST (1978) is the standard PORTMANTEAU test that pools the first h
autocorrelations into a single statistic and asks whether they are jointly consistent with white noise:

    Q = n(n+2) * sum_{k=1}^{h} rho_k^2 / (n - k),

where rho_k is the sample autocorrelation at lag k and the (n+2)/(n-k) weighting is a small-sample
correction over the older Box-Pierce statistic Q* = n sum rho_k^2. Under the null of no autocorrelation Q
is approximately chi-squared with h degrees of freedom (minus the number of fitted model parameters, if
testing residuals of an ARMA(p,q): df = h - p - q). A large Q -- small p-value -- rejects white noise and
says structure remains.

This module computes the sample autocorrelations (reusing the repo's autocorrelation routine), the
Ljung-Box and Box-Pierce statistics, and their chi-squared p-values (reusing the repo's chi-squared
survival function), with an optional degrees-of-freedom adjustment for fitted parameters. It is validated:
on white noise Q is small and the p-value large (fails to reject); on a strongly autocorrelated AR(1)
series Q is large and the p-value tiny; a seasonal (periodic) signal is flagged at the seasonal lag; the
Ljung-Box statistic exceeds Box-Pierce (the small-sample correction inflates it); the statistic grows with
the number of lags tested when real autocorrelation is present; the degrees-of-freedom adjustment for
fitted parameters lowers the reported p-value; and results are deterministic. Pure stdlib; the
time-series-diagnostic companion to the Levinson-Durbin, Burg, CUSUM, and Mann-Kendall tools."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from levinson_durbin import autocorrelation
from kruskal_wallis import chi2_sf


def autocorrelations(x, max_lag):
    """Normalized sample autocorrelations rho_k = gamma_k / gamma_0 for k = 0..max_lag."""
    gamma = autocorrelation(x, max_lag)
    g0 = gamma[0]
    if g0 == 0:
        return [1.0] + [0.0] * max_lag
    return [g / g0 for g in gamma]


def box_pierce(x, lags=10):
    """Box-Pierce statistic Q* = n sum_{k=1}^{h} rho_k^2. Returns (Q, p_value)."""
    n = len(x)
    rho = autocorrelations(x, lags)
    q = n * sum(rho[k] ** 2 for k in range(1, lags + 1))
    return q, chi2_sf(q, lags)


def ljung_box(x, lags=10, model_df=0):
    """Ljung-Box test. Returns a dict with Q, df, p_value, and the per-lag autocorrelations.

    lags is the number of autocorrelations pooled; model_df is the number of fitted model parameters
    (p+q for an ARMA) subtracted from the degrees of freedom when testing residuals."""
    n = len(x)
    rho = autocorrelations(x, lags)
    q = n * (n + 2) * sum(rho[k] ** 2 / (n - k) for k in range(1, lags + 1))
    df = max(1, lags - model_df)
    return {
        "Q": q,
        "df": df,
        "p_value": chi2_sf(q, df),
        "autocorrelations": rho[1:lags + 1],
        "lags": lags,
    }


def is_white_noise(x, lags=10, alpha=0.05):
    """Convenience: True if the Ljung-Box test fails to reject white noise at level alpha."""
    return ljung_box(x, lags)["p_value"] > alpha
