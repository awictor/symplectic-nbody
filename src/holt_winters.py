"""Exponential smoothing: forecasting a time series by weighting recent history most.

The oldest and still one of the most widely-used forecasting families, exponential smoothing predicts
the future of a series by an average of its past that decays geometrically -- last week matters more
than last year. Three levels of the method handle increasingly structured data:

  SIMPLE exponential smoothing (SES) fits a series with no trend or seasonality: the level is updated
      as l_t = alpha*y_t + (1-alpha)*l_{t-1}, a weighted blend of the new observation and the old
      level, and the forecast is just the last level (a flat line). alpha in (0,1) sets the memory.
  HOLT'S linear method adds a TREND component b, updated alongside the level, so the forecast is a
      sloped line l + h*b for h steps ahead -- it extrapolates growth.
  HOLT-WINTERS adds a SEASONAL component: a repeating pattern of period m (12 for monthly data,
      say), updated multiplicatively or additively, so the forecast reinstates the seasonal shape on
      top of the trending level. This is the workhorse for demand, traffic, and any series with a
      calendar rhythm.

Each is a simple online recurrence -- O(1) memory per step, O(n) to fit -- yet captures level, trend,
and season together. This module implements all three (additive Holt-Winters), fits them to a series,
forecasts h steps ahead, and reports the in-sample one-step errors.

Validated against synthetic series with known structure: SES tracks a noisy constant to its mean and
forecasts a flat line; Holt recovers a linear trend and its h-step forecast lies on the extrapolated
line; Holt-Winters reproduces a trend-plus-seasonal series and its forecast reinstates the seasonal
pattern; a higher alpha reacts faster to a level shift; and the one-step forecast error is small for a
well-specified model. Pure stdlib; the forecasting companion to the Kalman filter and the
autocorrelation / spectral tools."""

from __future__ import annotations


def ses(y, alpha):
    """Simple exponential smoothing. Returns (levels, forecast_fn) where forecast_fn(h) predicts h
    steps past the end (a flat line at the last level)."""
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    if not y:
        raise ValueError("empty series")
    level = y[0]
    levels = [level]
    for t in range(1, len(y)):
        level = alpha * y[t] + (1 - alpha) * level
        levels.append(level)
    last = level

    def forecast(h):
        return [last] * h
    return levels, forecast


def holt(y, alpha, beta):
    """Holt's linear-trend method. Returns (levels, trends, forecast_fn) with a sloped forecast."""
    if not (0 < alpha < 1) or not (0 < beta < 1):
        raise ValueError("alpha, beta must be in (0, 1)")
    if len(y) < 2:
        raise ValueError("need at least 2 points")
    level = y[0]
    trend = y[1] - y[0]
    levels = [level]
    trends = [trend]
    for t in range(1, len(y)):
        prev_level = level
        level = alpha * y[t] + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
        levels.append(level)
        trends.append(trend)

    def forecast(h):
        return [level + (i + 1) * trend for i in range(h)]
    return levels, trends, forecast


def holt_winters(y, alpha, beta, gamma, period):
    """Additive Holt-Winters (level + trend + seasonal). Returns (levels, trends, seasonals,
    forecast_fn). period m is the seasonal cycle length."""
    if not all(0 < p < 1 for p in (alpha, beta, gamma)):
        raise ValueError("alpha, beta, gamma must be in (0, 1)")
    m = period
    n = len(y)
    if n < 2 * m:
        raise ValueError(f"need at least {2 * m} points for period {m}")

    # initialize level, trend, and seasonal indices from the first two seasons
    level = sum(y[:m]) / m
    # trend: average difference between the first and second season
    trend = sum((y[m + i] - y[i]) for i in range(m)) / (m * m)
    seasonals = [y[i] - level for i in range(m)]  # additive seasonal deviations

    levels = []
    trends = []
    for t in range(n):
        season = seasonals[t % m]
        if t == 0:
            levels.append(level)
            trends.append(trend)
            continue
        prev_level = level
        level = alpha * (y[t] - season) + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
        seasonals[t % m] = gamma * (y[t] - level) + (1 - gamma) * season
        levels.append(level)
        trends.append(trend)

    def forecast(h):
        out = []
        for i in range(h):
            season = seasonals[(n + i) % m]
            out.append(level + (i + 1) * trend + season)
        return out
    return levels, trends, list(seasonals), forecast


def one_step_errors(y, fitted_levels):
    """In-sample one-step forecast errors: y_t minus the level predicted from t-1."""
    return [y[t] - fitted_levels[t - 1] for t in range(1, len(y))]


def rmse(errors):
    if not errors:
        return 0.0
    return (sum(e * e for e in errors) / len(errors)) ** 0.5
