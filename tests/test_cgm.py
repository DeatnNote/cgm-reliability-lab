import numpy as np
import pandas as pd

from cgm import analysis as A


def test_clarke_zones_on_known_points():
    ref = np.array([100, 100, 50, 300, 60, 200])
    cgm = np.array([110, 150, 60, 50, 250, 120])
    assert list(A.clarke_zone(ref, cgm)) == ["A", "B", "A", "E", "E", "B"]
    assert list(A.clarke_zone([250], [150])) == ["D"]


def test_20_20_uses_absolute_band_below_100():
    assert list(A.within_20_20([80, 80, 200, 200], [99, 101, 239, 241])) == [True, False, True, False]


def test_cluster_bootstrap_resamples_whole_participants():
    df = pd.DataFrame({"participant": [0, 0, 0, 1], "x": [1.0, 1.0, 1.0, 5.0]})
    point, lo, hi = A.cluster_boot(df, lambda g: g["x"].mean(), B=200)
    assert point == 2.0 and lo >= 1.0 and hi <= 5.0 and hi > lo


def test_accuracy_is_worst_on_day_one_and_in_lows(d):
    a = A.accuracy(d, B=50)
    assert a["mard_by_day"][1] > a["mard_by_day"][4]
    assert a["by_range"]["<70"]["mard"] > a["by_range"]["70-180"]["mard"]
    assert a["clarke"]["A"] > 0.9 and a["clarke"]["E"] == 0


def test_firmware_v2_trades_noise_for_delay(d):
    f = A.firmware(d, B=50)
    assert f["delay_minutes"]["v2"] < f["delay_minutes"]["v1"]
    assert f["fast_change_over_1_mg_dl_min"]["diff_v2_minus_v1"][2] < 0       # better when glucose moves fast
    assert f["noise_second_difference"]["v2"] > f["noise_second_difference"]["v1"]


def test_bench_model_finds_the_line2_offset(d):
    b = A.bench_to_body(d)
    assert b["line2_offset_ci"][0] > 0 and b["r2_bench_plus_line"] > b["r2_bench_only"]


def test_reliability_names_the_planted_causes(d):
    r = A.reliability(d)
    cox = r["cox_cause_specific"]
    assert cox["detached"]["adh7"]["lo"] > 3                                 # the bad adhesive batch
    assert cox["sensor_failure"]["tenths_um_below_9_4"]["lo"] > 1.2           # thin membranes
    assert cox["sensor_failure"]["adh7"]["lo"] < 1 < cox["sensor_failure"]["adh7"]["hi"] or cox["sensor_failure"]["adh7"]["hi"] < 1.3


def test_compression_model_is_safe_and_better_than_the_rule(d):
    c = A.compression(d)
    assert c["model"]["true_lows_suppressed"] <= 3
    assert c["model"]["compression_alarms_suppressed"] > c["rule_rebound_over_2"]["compression_alarms_suppressed"]


def test_real_world_truth_is_consistent(d):
    rw = d["real_world"]
    assert rw["days_worn"].between(0, 15).all()
    assert (rw.loc[rw["end_reason"] == "completed", "days_worn"] == 15).all()
