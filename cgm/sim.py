"""Simulated CGM program with known truth.

Manufacturing: 48 sensor lots (2024-2025) on two lines, each with a membrane thickness, an enzyme loading and an
adhesive batch. Hidden effects (known only here):
  - sensitivity: thicker membranes lower the sensor's in-body sensitivity (readings biased low); enzyme loading
    raises it slightly;
  - early failures: one adhesive batch ("ADH-7") detaches early; membranes thinner than about 9.3 um fail early
    from signal loss.
Data sources:
  - bench:      20 sensors per lot tested in buffer: in-vitro sensitivity (noisy, partly predictive of in-body bias);
  - clinical:   720 participants on 36 lots, 15-day wear, lab reference glucose every 15 min during 5 clinic days;
  - real world: 6,000 sessions on all 48 lots: wear duration and why it ended (failure, detachment, completed).
Sensor model: interstitial lag (~9 min), lot gain, a first-day bias that fades, slow drift over the wear, and noise.
Raw sensor current is smoothed by firmware; v1 = exponential smoothing, v2 = a lag-compensated filter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STEP = 5                     # minutes between sensor readings
DAYS = 15
N_PTS = DAYS * 24 * 60 // STEP
CLINIC_DAYS = (1, 4, 8, 12, 15)


def lots_table(n: int = 48, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    month = pd.period_range("2024-01", periods=24, freq="M")
    thick = rng.normal(10.0, 0.55, n)
    thick[[9, 21, 33, 40]] -= 0.9                                 # a few lots drift thin
    lot = pd.DataFrame({"lot": [f"L{i:03d}" for i in range(n)], "made": [str(month[i // 2]) for i in range(n)],
                        "line": rng.choice(["Line-1", "Line-2"], n), "membrane_um": thick.round(2),
                        "enzyme_u": rng.normal(100, 6, n).round(1),
                        "adhesive": rng.choice([f"ADH-{k}" for k in range(1, 10)], n)})
    lot.loc[[14, 15, 16, 26, 27], "adhesive"] = "ADH-7"
    # hidden truth: in-body sensitivity ratio (1 = perfectly calibrated factory code)
    lot["true_gain"] = 1 - 0.06 * (lot["membrane_um"] - 10) + 0.0012 * (lot["enzyme_u"] - 100) + rng.normal(0, 0.015, n)
    return lot


def bench(lots: pd.DataFrame, per_lot: int = 20, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for r in lots.itertuples():
        # buffer sensitivity tracks the in-body gain, but with lot-level bench bias and sensor noise
        lot_bench = r.true_gain + rng.normal(0, 0.015) - 0.02 * (r.line == "Line-2")
        for k in range(per_lot):
            rows.append((r.lot, k, round(float(lot_bench + rng.normal(0, 0.03)), 4)))
    return pd.DataFrame(rows, columns=["lot", "sensor", "bench_sensitivity_ratio"])


def glucose(n: int, rng, pts: int = N_PTS) -> np.ndarray:
    """True blood glucose (mg/dL), every 5 minutes: personal mean, daily rhythm, meals, random walk."""
    t = np.arange(pts) * STEP / 60.0                               # hours
    mean = rng.normal(138, 25, (n, 1))
    g = mean + 15 * np.sin(2 * np.pi * (t - 4) / 24)[None, :]
    for meal_h in (7.5, 12.5, 19.0):
        for d in range(DAYS):
            start = d * 24 + meal_h + rng.normal(0, 0.6, (n, 1))
            dt = t[None, :] - start
            amp = rng.uniform(25, 85, (n, 1))
            g = g + np.where(dt > 0, amp * (dt / 1.2) * np.exp(1 - dt / 1.2), 0)
    walk = np.cumsum(rng.normal(0, 2.2, (n, pts)), 1)
    walk = walk - pd.DataFrame(walk.T).rolling(144, min_periods=1).mean().to_numpy().T   # remove slow wander
    g = g + walk
    # hypoglycaemia episodes (a few per person over the wear)
    for _ in range(8):
        c = rng.integers(0, pts, (n, 1))
        depth = rng.uniform(50, 120, (n, 1))
        width = rng.uniform(4, 12, (n, 1))                          # in 5-min steps
        g = g - depth * np.exp(-0.5 * ((np.arange(pts)[None, :] - c) / width) ** 2) * (rng.random((n, 1)) < 0.8)
    return np.clip(g, 40, 400)


def ema(x: np.ndarray, alpha: float) -> np.ndarray:
    y = np.empty_like(x)
    y[:, 0] = x[:, 0]
    for i in range(1, x.shape[1]):
        y[:, i] = alpha * x[:, i] + (1 - alpha) * y[:, i - 1]
    return y


def sensor(true_g: np.ndarray, gain: np.ndarray, rng, firmware: str = "v1") -> tuple[np.ndarray, np.ndarray]:
    """Returns (raw, displayed) sensor glucose. gain: per-sensor gain (lot gain x sensor variation)."""
    n, pts = true_g.shape
    lag = 2                                                        # ~10 min interstitial delay
    isf = np.concatenate([np.repeat(true_g[:, :1], lag, 1), true_g[:, :-lag]], 1)
    day = np.arange(pts) * STEP / 1440.0
    first_day = -0.12 * np.exp(-day / 0.5)                          # reads low on day 1, fading over ~a day
    drift = -0.004 * day                                            # slow sensitivity loss
    g_t = gain[:, None] * (1 + first_day[None, :] + drift[None, :])
    raw = isf * g_t + rng.normal(0, 1, (n, pts)) * (5 + 0.04 * isf)
    return raw, smooth(raw, firmware)


def smooth(raw: np.ndarray, firmware: str) -> np.ndarray:
    if firmware == "v1":
        return ema(raw, 0.45)
    # v2: smoother, plus a trend term that compensates the lag both the filter and the interstitium add
    s = ema(raw, 0.30)
    trend = np.diff(s, axis=1, prepend=s[:, :1])
    return s + 2.6 * ema(trend, 0.4)


def clinical(lots: pd.DataFrame, n_per_lot: int = 20, n_lots: int = 36, seed: int = 2) -> dict:
    rng = np.random.default_rng(seed)
    use = lots.sample(n_lots, random_state=seed)["lot"].to_numpy()
    lot_of = np.repeat(use, n_per_lot)
    n = len(lot_of)
    g = glucose(n, rng)
    gain = lots.set_index("lot").loc[lot_of, "true_gain"].to_numpy() * rng.normal(1, 0.035, n)
    raw, v1 = sensor(g, gain, rng, "v1")
    v2 = smooth(raw, "v2")
    # reference: lab analyser every 15 min for 12 h on each clinic day, with its own small error
    idx = []
    for d in CLINIC_DAYS:
        start = (d - 1) * 288 + (24 if d == 1 else 96)             # day 1 starts 2 h after insertion, others 08:00
        idx += list(range(start, start + 144, 3))
    idx = np.array(idx)
    ref = g[:, idx] * (1 + rng.normal(0, 0.015, (n, len(idx))))
    pid = np.repeat(np.arange(n), len(idx))
    pts = pd.DataFrame({"participant": pid, "lot": np.repeat(lot_of, len(idx)), "minute": np.tile(idx * STEP, n),
                        "day": np.tile(idx * STEP // 1440 + 1, n), "reference": ref.ravel().round(1),
                        "cgm_v1": np.clip(v1[:, idx], 40, 400).ravel().round(1), "cgm_v2": np.clip(v2[:, idx], 40, 400).ravel().round(1),
                        "true_glucose": g[:, idx].ravel().round(1)})
    parts = pd.DataFrame({"participant": np.arange(n), "lot": lot_of, "sensor_gain": gain.round(4),
                          "diabetes_type": rng.choice(["T1D", "T2D"], n, p=[0.6, 0.4])})
    return {"pairs": pts, "participants": parts, "_glucose": g, "_raw": raw, "_v1": v1, "_v2": v2}


def real_world(lots: pd.DataFrame, n: int = 6000, seed: int = 3) -> pd.DataFrame:
    """One row per sensor session: how long it lasted and why it ended (Weibull hazards, known causes)."""
    rng = np.random.default_rng(seed)
    lt = lots.set_index("lot")
    lot = rng.choice(lots["lot"], n)
    fw = np.where(pd.PeriodIndex(lt.loc[lot, "made"], freq="M") >= pd.Period("2025-01", "M"), "v2", "v1")
    thick = lt.loc[lot, "membrane_um"].to_numpy()
    adh7 = (lt.loc[lot, "adhesive"] == "ADH-7").to_numpy()
    site = rng.choice(["abdomen", "arm"], n, p=[0.45, 0.55])
    active = rng.random(n) < 0.3                                    # sports / swimming
    # failure (signal loss): thin membranes; detachment: adhesive batch, activity
    def weib(scale, k=1.6):
        return scale * rng.weibull(k, n)
    t_fail = weib(80 * np.exp(-2.2 * np.clip(9.4 - thick, 0, None)) * np.exp(-0.15 * (thick < 9.4)))
    t_det = weib(120 / (1 + 3.5 * adh7) / (1 + 0.8 * active) / np.where(site == "abdomen", 1.3, 1.0), 1.3)
    t_user = weib(200, 1.0)                                         # removed for other reasons (censoring)
    end = np.minimum.reduce([t_fail, t_det, t_user, np.full(n, float(DAYS))])
    reason = np.select([end == t_fail, end == t_det, end == t_user], ["sensor_failure", "detached", "removed_by_user"], "completed")
    return pd.DataFrame({"session": np.arange(n), "lot": lot, "firmware": fw, "site": site, "active_user": active,
                         "days_worn": end.round(2), "end_reason": reason})


def night_windows(n: int = 4000, seed: int = 4) -> pd.DataFrame:
    """3-hour overnight windows of displayed CGM (5-min readings) with what really happened:
       'compression' (sensor pressed while sleeping: false low), 'true_low' (real hypoglycaemia), 'normal'."""
    rng = np.random.default_rng(seed)
    pts = 36
    kind = rng.choice(["normal", "compression", "true_low"], n, p=[0.5, 0.3, 0.2])
    base = rng.normal(130, 25, n)
    t = np.arange(pts)
    g = base[:, None] + np.cumsum(rng.normal(0, 1.5, (n, pts)), 1)
    start = rng.integers(6, 22, n)
    dur = rng.integers(4, 16, n)                                    # 20-80 min
    for i in np.where(kind == "true_low")[0]:                       # fall to a low; recovery fast if treated with carbs
        fall = np.clip((t - start[i]) / rng.uniform(2.5, 10), 0, 1)
        rec = np.clip((t - start[i] - dur[i] - 4) / rng.uniform(2, 10), 0, 1)
        target = rng.uniform(45, 68)
        g[i] = g[i] - (g[i] - target) * fall * (1 - rec)
    true_g = g.copy()
    s = g.copy()
    for i in np.where(kind == "compression")[0]:                    # pressure on the sensor: drop, plateau, release
        depth = rng.uniform(0.3, 0.6)
        up = np.clip((t - start[i] + 1) / rng.uniform(1, 6), 0, 1)
        down = 1 - np.clip((t - start[i] - dur[i]) / rng.uniform(1, 6), 0, 1)
        s[i] = s[i] * (1 - depth * up * down)
    s = s + rng.normal(0, 4.5, s.shape)
    df = pd.DataFrame(s.round(1), columns=[f"t{j:02d}" for j in t])
    df.insert(0, "kind", kind)
    df.insert(1, "session", rng.integers(0, 2000, n))
    df["true_min"] = true_g.min(1).round(1)
    return df


def simulate(seed: int = 0) -> dict:
    lots = lots_table(seed=seed)
    return {"lots": lots, "bench": bench(lots, seed=seed + 1), "clinical": clinical(lots, seed=seed + 2),
            "real_world": real_world(lots, seed=seed + 3), "night": night_windows(seed=seed + 4)}
