"""Every analysis in the lab. Confidence intervals resample PARTICIPANTS (or lots), because readings from the same
sensor are correlated."""
from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng


# ------------------------------------------------------------------ accuracy against the lab reference
def clarke_zone(ref: np.ndarray, cgm: np.ndarray) -> np.ndarray:
    """Clarke error grid zone per pair (A: clinically accurate ... E: opposite treatment)."""
    r, c = np.asarray(ref, float), np.asarray(cgm, float)
    z = np.full(len(r), "B", dtype=object)
    A = ((r < 70) & (c < 70)) | (np.abs(c - r) <= 0.2 * r)
    E = ((r >= 180) & (c <= 70)) | ((r <= 70) & (c >= 180))
    C = ((r >= 70) & (r <= 290) & (c >= r + 110)) | ((r >= 130) & (r <= 180) & (c <= 7 / 5 * r - 182))
    D = ((r >= 240) & (c >= 70) & (c <= 180)) | ((r <= 175 / 3) & (c <= 180) & (c >= 70)) | ((r >= 175 / 3) & (r <= 70) & (c >= 6 / 5 * r))
    z[D], z[C], z[E], z[A] = "D", "C", "E", "A"
    return z


def within_20_20(ref, cgm) -> np.ndarray:
    ref, cgm = np.asarray(ref), np.asarray(cgm)
    return np.where(ref < 100, np.abs(cgm - ref) <= 20, np.abs(cgm - ref) <= 0.2 * ref)


def cluster_boot(df: pd.DataFrame, stat, by: str = "participant", B: int = 500, seed: int = 0):
    rng = RNG(seed)
    ids = df[by].unique()
    groups = {k: v for k, v in df.groupby(by).indices.items()}
    point = stat(df)
    vals = []
    for _ in range(B):
        pick = rng.choice(ids, len(ids))
        vals.append(stat(df.iloc[np.concatenate([groups[k] for k in pick])]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def mard(col):
    return lambda d: float(((d[col] - d["reference"]).abs() / d["reference"]).mean())


def accuracy(d: dict, col: str = "cgm_v1", B: int = 500) -> dict:
    p = d["clinical"]["pairs"].merge(d["clinical"]["participants"][["participant", "diabetes_type"]], on="participant")
    out = {"participants": int(p["participant"].nunique()), "pairs": int(len(p)), "mard": cluster_boot(p, mard(col), B=B)}
    out["within_20_20"] = cluster_boot(p, lambda x: float(within_20_20(x["reference"], x[col]).mean()), B=B)
    z = pd.Series(clarke_zone(p["reference"], p[col])).value_counts(normalize=True)
    out["clarke"] = {k: float(z.get(k, 0)) for k in "ABCDE"}
    out["mard_by_day"] = {int(k): mard(col)(g) for k, g in p.groupby("day")}
    rng = pd.cut(p["reference"], [0, 70, 180, 1000], labels=["<70", "70-180", ">180"])
    out["by_range"] = {str(k): {"pairs": int(len(g)), "mard": mard(col)(g), "mad_mg_dl": float((g[col] - g["reference"]).abs().mean())}
                       for k, g in p.groupby(rng, observed=True)}
    out["by_diabetes_type"] = {k: cluster_boot(g, mard(col), B=B // 2) for k, g in p.groupby("diabetes_type")}
    return out


# ------------------------------------------------------------------ firmware v1 vs v2, paired on the same sensors
def firmware(d: dict, B: int = 500) -> dict:
    p = d["clinical"]["pairs"].sort_values(["participant", "minute"]).copy()
    # rate of change from the REFERENCE series (what a study can actually measure), mg/dL per minute
    same = p["participant"].eq(p["participant"].shift()) & p["day"].eq(p["day"].shift())
    p["roc"] = np.where(same, (p["reference"] - p["reference"].shift()) / 15, np.nan)
    p["fast"] = p["roc"].abs() > 1.0
    diff = lambda g: mard("cgm_v2")(g) - mard("cgm_v1")(g)            # noqa: E731
    mad = lambda c: (lambda g: float((g[c] - g["reference"]).abs().mean()))  # noqa: E731
    lows = p[p["reference"] < 70]
    out = {"overall": {"v1": mard("cgm_v1")(p), "v2": mard("cgm_v2")(p), "diff_v2_minus_v1": cluster_boot(p, diff, B=B)},
           "fast_change_over_1_mg_dl_min": {"pairs": int(p["fast"].sum()), "v1": mard("cgm_v1")(p[p["fast"]]), "v2": mard("cgm_v2")(p[p["fast"]]),
                                            "diff_v2_minus_v1": cluster_boot(p[p["fast"]], diff, B=B)},
           "stable": {"v1": mard("cgm_v1")(p[~p["fast"]]), "v2": mard("cgm_v2")(p[~p["fast"]]),
                      "diff_v2_minus_v1": cluster_boot(p[~p["fast"]], diff, B=B)},
           "below_70_mad_mg_dl": {"pairs": int(len(lows)), "v1": mad("cgm_v1")(lows), "v2": mad("cgm_v2")(lows),
                                  "diff_v2_minus_v1": cluster_boot(lows, lambda g: mad("cgm_v2")(g) - mad("cgm_v1")(g), B=B)}}
    # delay: lag (minutes) maximizing correlation between true glucose and the displayed trace
    g, v1, v2 = d["clinical"]["_glucose"][:100], d["clinical"]["_v1"][:100], d["clinical"]["_v2"][:100]
    def lag(x):
        best = max(range(0, 9), key=lambda k: np.corrcoef(g[:, : g.shape[1] - k].ravel(), x[:, k:].ravel())[0, 1])
        return best * 5
    out["delay_minutes"] = {"v1": lag(v1), "v2": lag(v2)}
    noise = lambda x: float(np.std(np.diff(x, 2, axis=1)))           # noqa: E731
    out["noise_second_difference"] = {"v1": noise(v1), "v2": noise(v2)}
    return out


# ------------------------------------------------------------------ bench -> in-body bias, and a lot-release rule
def bench_to_body(d: dict, limit: float = 0.05, p_hold: float = 0.2) -> dict:
    """Can bench (in-vitro) sensitivity predict a lot's in-body bias before any clinical data exist?
    Bias is a lot's deviation from the typical lot (factory calibration is set for the typical lot); a lot is out
    of spec beyond +-limit. Every lot is scored: the 36 clinical lots leave-one-out, the other 12 from the full fit."""
    import statsmodels.formula.api as smf
    from scipy.stats import norm
    bm = d["bench"].groupby("lot")["bench_sensitivity_ratio"].mean().rename("bench_raw")
    p = d["clinical"]["pairs"]
    p = p[p["day"].between(4, 12)]
    per = (p["cgm_v1"] / p["reference"]).groupby(p["lot"]).median().rename("clinical_ratio")
    L = d["lots"].set_index("lot").join(bm).join(per)
    L["line2"] = (L["line"] == "Line-2").astype(int)
    L["bench"] = L["bench_raw"] / L["bench_raw"].median() - 1
    L["bias"] = L["clinical_ratio"] / L["clinical_ratio"].median() - 1
    L["true_dev"] = L["true_gain"] / L["true_gain"].median() - 1
    L["out_of_spec"] = L["true_dev"].abs() > limit
    clin = L[L["bias"].notna()]
    m0, m1 = smf.ols("bias ~ bench", clin).fit(), smf.ols("bias ~ bench + line2", clin).fit()
    def predict(fit, rows):
        f = fit.get_prediction(rows).summary_frame()
        sd = np.sqrt(f["mean_se"] ** 2 + fit.scale)
        return f["mean"].to_numpy(), sd.to_numpy()
    mu, sd = np.zeros(len(L)), np.zeros(len(L))
    for i, k in enumerate(L.index):
        fit = smf.ols("bias ~ bench + line2", clin.drop(k)).fit() if k in clin.index else m1
        mu[i], sd[i] = (x[0] for x in predict(fit, L.loc[[k]]))
    L["pred"], L["pred_sd"] = mu, sd
    L["p_out"] = norm.cdf(-limit, mu, sd) + norm.sf(limit, mu, sd)
    L["hold_model"] = L["p_out"] > p_hold
    L["hold_naive"] = L["bench"].abs() > limit                     # bench result alone, no line correction
    def score(h):
        return {"held": int(h.sum()), "out_of_spec_caught": int((h & L["out_of_spec"]).sum()),
                "out_of_spec_missed": int((~h & L["out_of_spec"]).sum()), "good_lots_held": int((h & ~L["out_of_spec"]).sum())}
    loo_err = L.loc[clin.index, "pred"] - L.loc[clin.index, "bias"]
    return {"lots": int(len(L)), "lots_with_clinical_data": int(len(clin)), "out_of_spec_lots": int(L["out_of_spec"].sum()),
            "limit": limit, "r2_bench_only": float(m0.rsquared), "r2_bench_plus_line": float(m1.rsquared),
            "line2_offset": float(m1.params["line2"]), "line2_offset_ci": [float(x) for x in m1.conf_int().loc["line2"]],
            "slope": float(m1.params["bench"]), "slope_ci": [float(x) for x in m1.conf_int().loc["bench"]],
            "loo_rmse": float(np.sqrt(np.mean(loo_err ** 2))),
            "release_model": score(L["hold_model"]), "release_naive_bench": score(L["hold_naive"]),
            "table": L[["line", "membrane_um", "bench", "bias", "pred", "p_out", "hold_model", "hold_naive", "true_dev", "out_of_spec"]]
                     .round(4).reset_index().to_dict("records")}


# ------------------------------------------------------------------ reliability: survival, causes, a tolerance
def reliability(d: dict) -> dict:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    rw = d["real_world"].merge(d["lots"], on="lot")
    rw["adh7"] = (rw["adhesive"] == "ADH-7").astype(int)
    rw["line2"] = (rw["line"] == "Line-2").astype(int)
    rw["arm"] = (rw["site"] == "arm").astype(int)
    rw["active"] = rw["active_user"].astype(int)
    rw["fw_v2"] = (rw["firmware"] == "v2").astype(int)
    out = {"sessions": int(len(rw)), "end_reasons": rw["end_reason"].value_counts(normalize=True).round(4).to_dict(),
           "full_wear_rate": float((rw["days_worn"] >= 15).mean())}
    km = {}
    for name, g in (("ADH-7 adhesive", rw[rw["adh7"] == 1]), ("other adhesives", rw[rw["adh7"] == 0]),
                    ("membrane < 9.4 um", rw[rw["membrane_um"] < 9.4]), ("membrane >= 9.4 um", rw[rw["membrane_um"] >= 9.4])):
        k = KaplanMeierFitter().fit(g["days_worn"], g["end_reason"].isin(["sensor_failure", "detached"]))
        km[name] = {"sessions": int(len(g)), "survival_day_10": float(k.predict(10.0)), "survival_day_15": float(k.predict(14.99))}
    out["kaplan_meier_any_early_end"] = km
    # membrane enters as a hinge, "um below 9.4": failure rates by thickness band are flat above ~9.4 um and climb
    # steeply below it (see membrane_bands). Lot-level factors are shared by every session from a lot, so standard
    # errors are clustered by lot.
    rw["tenths_um_below_9_4"] = 10 * (9.4 - rw["membrane_um"]).clip(lower=0)      # HR per 0.1 um thinner than 9.4
    X = ["tenths_um_below_9_4", "adh7", "line2", "arm", "active", "fw_v2"]
    cox = {}
    for cause in ("sensor_failure", "detached"):
        df = rw[X + ["days_worn", "lot"]].assign(event=rw["end_reason"].eq(cause).astype(int))
        c = CoxPHFitter(penalizer=0.01).fit(df, "days_worn", "event", cluster_col="lot")   # light ridge: steep hinge
        s = c.summary
        cox[cause] = {x: {"hr": float(s.loc[x, "exp(coef)"]), "lo": float(s.loc[x, "exp(coef) lower 95%"]),
                          "hi": float(s.loc[x, "exp(coef) upper 95%"]), "p": float(s.loc[x, "p"])} for x in X}
    out["cox_cause_specific"] = cox
    bands = pd.cut(rw["membrane_um"], [0, 9.0, 9.4, 9.8, 10.2, 99], labels=["<9.0", "9.0-9.4", "9.4-9.8", "9.8-10.2", ">=10.2"], right=False)
    out["membrane_bands"] = {}
    for b, g in rw.groupby(bands, observed=True):
        k = KaplanMeierFitter().fit(g["days_worn"], g["end_reason"].eq("sensor_failure"))
        out["membrane_bands"][str(b)] = {"sessions": int(len(g)), "lots": int(g["lot"].nunique()), "failure_by_day_15": float(1 - k.predict(14.99))}
    # tolerance: choose the minimum membrane thickness on 2024 lots, check it on 2025 lots
    rw["early_fail"] = (rw["end_reason"] == "sensor_failure") & (rw["days_worn"] < 10)
    tr, te = rw[rw["made"] < "2025-01"], rw[rw["made"] >= "2025-01"]
    cands = np.round(np.arange(9.0, 10.01, 0.1), 1)
    def effect(df, cut):
        keep = df["membrane_um"] >= cut
        return float(df["early_fail"].mean()), float(df.loc[keep, "early_fail"].mean()), float(1 - keep.mean())
    scores = {c: effect(tr, c) for c in cands}
    best = min(cands, key=lambda c: scores[c][1] + 0.5 * scores[c][2])   # trade failures against scrapped lots
    out["membrane_tolerance"] = {"chosen_on_2024_lots_um": float(best),
                                 "2024": dict(zip(("early_failure_rate_all", "early_failure_rate_kept", "share_of_sessions_scrapped"), scores[best])),
                                 "2025_check": dict(zip(("early_failure_rate_all", "early_failure_rate_kept", "share_of_sessions_scrapped"), effect(te, best)))}
    # naive view: share ended early per lot, ignoring censoring and cause
    naive = rw.groupby("lot").apply(lambda g: (g["end_reason"] != "completed").mean(), include_groups=False)
    out["worst_lots_naive"] = naive.sort_values(ascending=False).head(6).round(3).to_dict()
    out["worst_lots_naive_adhesive"] = d["lots"].set_index("lot").loc[list(out["worst_lots_naive"]), "adhesive"].to_dict()
    return out


# ------------------------------------------------------------------ compression lows vs true lows overnight
def night_features(nw: pd.DataFrame) -> pd.DataFrame:
    x = nw[[c for c in nw.columns if c.startswith("t")]].to_numpy()
    dx = np.diff(x, axis=1) / 5                                     # mg/dL per minute
    mi = x.argmin(1)
    f = pd.DataFrame({"min": x.min(1), "start": x[:, :4].mean(1), "max_fall": -dx.min(1), "max_rise": dx.max(1),
                      "drop_frac": 1 - x.min(1) / x[:, :4].mean(1), "time_below_70": (x < 70).sum(1) * 5,
                      "fall_before_min": [(x[i, max(0, m - 3)] - x[i, m]) / 15 for i, m in enumerate(mi)],
                      "rise_after_min": [(x[i, min(x.shape[1] - 1, m + 3)] - x[i, m]) / 15 for i, m in enumerate(mi)],
                      "roughness": np.abs(np.diff(x, 2, axis=1)).mean(1)})
    return f


def compression(d: dict, seed: int = 0) -> dict:
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    nw = d["night"]
    f = night_features(nw)
    alarm = f["min"] < 70                                           # windows where the CGM would sound a low alarm
    y = (nw["kind"] == "compression").astype(int)
    df = f[alarm].assign(y=y[alarm], kind=nw.loc[alarm, "kind"], session=nw.loc[alarm, "session"])
    tr, te = df[df["session"] % 3 != 0], df[df["session"] % 3 == 0]  # split by sensor session
    X = list(f.columns)
    from sklearn.model_selection import GroupKFold, cross_val_predict
    mk = lambda: LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=seed)  # noqa: E731
    # threshold from OUT-OF-FOLD training predictions (in-sample ones are overconfident):
    # suppress an alarm only where <= 1% of true lows would have been suppressed
    oof = cross_val_predict(mk(), tr[X], tr["y"], cv=GroupKFold(5), groups=tr["session"], method="predict_proba")[:, 1]
    t = float(np.quantile(oof[(tr["kind"] == "true_low").to_numpy()], 0.99))
    m = mk().fit(tr[X], tr["y"])
    p_te = m.predict_proba(te[X])[:, 1]
    sup = p_te > t
    rule = te["rise_after_min"] > 2.0                               # "bounced back faster than 2 mg/dL/min" rule
    tl, cp = te["kind"] == "true_low", te["kind"] == "compression"
    return {"alarm_windows": int(alarm.sum()), "test_alarms": int(len(te)), "test_true_lows": int(tl.sum()), "test_compressions": int(cp.sum()),
            "auc_model": float(roc_auc_score(te["y"], p_te)), "auc_rule_rebound": float(roc_auc_score(te["y"], te["rise_after_min"])),
            "model": {"compression_alarms_suppressed": float(sup[cp].mean()), "true_lows_suppressed": int((sup & tl).sum())},
            "rule_rebound_over_2": {"compression_alarms_suppressed": float(rule[cp].mean()), "true_lows_suppressed": int((rule & tl).sum())},
            "false_alarm_share_before": float(cp.mean()), "false_alarm_share_after_model": float((cp & ~sup).sum() / max((~sup).sum(), 1))}
