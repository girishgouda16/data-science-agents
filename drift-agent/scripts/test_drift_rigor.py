"""Drift rigour: small samples do not cry wolf (a PSI noise floor), a feed
going null and a column changing type are caught, timestamps and ids are
not "drift", a joint shift no single column shows is caught by the domain
classifier, drift over time names when it started, a brand-new category is
flagged, and any file format reads.
No pytest fixtures — CI also runs this file as a plain script.

Run: python test_drift_rigor.py
"""
import json
import tempfile as _tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import mcp_server as m


def _files(tmp_path, ref, cur, ref_name="ref.csv", cur_name="cur.csv"):
    ref_path, cur_path = tmp_path / ref_name, tmp_path / cur_name
    for df, path in ((ref, ref_path), (cur, cur_path)):
        df.to_parquet(path, index=False) if path.suffix == ".parquet" else df.to_csv(path, index=False)
    return str(ref_path), str(cur_path)


def test_small_samples_do_not_cry_wolf_but_real_shifts_still_show(tmp_path):
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"usage": rng.normal(0, 1, 5000)})
    same = pd.DataFrame({"usage": rng.normal(0, 1, 120)})
    moved = pd.DataFrame({"usage": rng.normal(1.0, 1, 120)})
    ref_path, same_path = _files(tmp_path, ref, same)
    quiet = json.loads(m.detect_drift(ref_path, same_path))
    col = quiet["columns"]["usage"]
    assert col["severity"] == "none" and col["psi_noise_floor"] > 0.1, col
    assert "small_sample_warning" in quiet and quiet["overall_severity"] == "none"
    _, moved_path = _files(tmp_path, ref, moved, cur_name="moved.csv")
    assert json.loads(m.detect_drift(ref_path, moved_path))["columns"]["usage"]["severity"] == "severe"


def test_a_feed_going_null_and_a_type_change_are_caught(tmp_path):
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"data_mb": rng.lognormal(5, 1, 3000), "arpu": rng.normal(20, 5, 3000)})
    cur = pd.DataFrame({"data_mb": rng.lognormal(5, 1, 3000), "arpu": rng.normal(20, 5, 3000).round(2).astype(str)})
    cur.loc[cur.sample(frac=0.4, random_state=1).index, "data_mb"] = np.nan  # a broken join upstream
    cur["arpu"] = cur["arpu"].str.replace(".", ",", regex=False)  # a locale change in the export
    result = json.loads(m.detect_drift(*_files(tmp_path, ref, cur)))
    data = result["columns"]["data_mb"]
    assert data["severity"] == "severe" and data["current_missing_pct"] > 35, data
    assert "data_mb" in result["missingness_changes"] and result["type_changes"] == {"arpu": "numeric -> text"}
    assert result["overall_severity"] == "severe"


def test_timestamps_and_identifiers_are_not_drift(tmp_path):
    rng = np.random.default_rng(2)
    n = 2000
    ref = pd.DataFrame({"ts": pd.date_range("2026-01-01", periods=n, freq="h").astype(str),
                        "msisdn": [f"4478{i:08d}" for i in range(n)], "calls": rng.poisson(5, n)})
    cur = pd.DataFrame({"ts": pd.date_range("2026-06-01", periods=n, freq="h").astype(str),
                        "msisdn": [f"4479{i:08d}" for i in range(n)], "calls": rng.poisson(5, n)})
    result = json.loads(m.detect_drift(*_files(tmp_path, ref, cur)))
    assert set(result["excluded_structural_columns"]) == {"ts", "msisdn"}, result["excluded_structural_columns"]
    assert result["overall_severity"] == "none", result["drifted_columns"]


def test_a_joint_shift_only_the_domain_classifier_sees(tmp_path):
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 4000)
    ref = pd.DataFrame({"voice": x[:2000], "data": 0.9 * x[:2000] + rng.normal(0, 0.44, 2000)})
    cur = pd.DataFrame({"voice": x[2000:], "data": -0.9 * x[2000:] + rng.normal(0, 0.44, 2000)})
    ref_path, cur_path = _files(tmp_path, ref, cur)
    assert json.loads(m.detect_drift(ref_path, cur_path))["overall_severity"] == "none"  # marginals unchanged
    joint = json.loads(m.detect_multivariate_drift(ref_path, cur_path))
    assert joint["severity"] == "severe" and joint["auc"] > 0.9, joint
    assert {c["column"] for c in joint["top_columns"]} == {"voice", "data"}
    same = pd.DataFrame({"voice": x[2000:], "data": 0.9 * x[2000:] + rng.normal(0, 0.44, 2000)})
    _, same_path = _files(tmp_path, ref, same, cur_name="same.csv")
    assert json.loads(m.detect_multivariate_drift(ref_path, same_path))["severity"] == "none"


def test_drift_over_time_names_the_week_it_started(tmp_path):
    rng = np.random.default_rng(4)
    ref = pd.DataFrame({"latency": rng.normal(40, 5, 5000), "drops": rng.poisson(1, 5000)})
    days = pd.date_range("2026-03-02", periods=56, freq="D")  # eight Monday-start weeks
    cur = pd.concat([pd.DataFrame({"day": d, "latency": rng.normal(40 + (12 if d >= days[28] else 0), 5, 300),
                                   "drops": rng.poisson(1, 300)}) for d in days])
    result = json.loads(m.drift_over_time(*_files(tmp_path, ref, cur), time_column="day", freq="W"))
    assert result["periods"] == 8 and result["first_drift"]["columns"] == ["latency"], result["first_drift"]
    assert result["first_drift"]["period"].startswith("2026-03-30"), result["first_drift"]
    assert [p["severity"] for p in result["timeline"]][:4] == ["none"] * 4 and result["latest_severity"] == "severe"


def test_a_new_category_is_flagged_even_when_psi_is_small(tmp_path):
    rng = np.random.default_rng(5)
    ref = pd.DataFrame({"device": rng.choice(["ios", "android"], 5000)})
    cur = pd.DataFrame({"device": np.where(rng.random(5000) < 0.03, "kaios", rng.choice(["ios", "android"], 5000))})
    col = json.loads(m.detect_drift(*_files(tmp_path, ref, cur, "ref.parquet")))["columns"]["device"]
    assert col["new_categories"] == ["kaios"] and col["severity"] == "moderate", col


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            with _tempfile.TemporaryDirectory() as d:
                fn(Path(d))
            print(f"ok {name}")
    print("all drift rigour checks passed")
