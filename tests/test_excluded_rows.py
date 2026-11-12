import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
import final, excluded_rows as ER
from utils import load_config

CFG = load_config()

def test_groups_partition_the_test_rows(fake_features, tmp_path):
    tp = final.fit_and_score(fake_features, CFG, {"ridge": {"alpha": 5.0}}, "ridge", models=("linear", "ridge"), out_dir=tmp_path / "a",
                             processed_dir=tmp_path / "p", lock_path=tmp_path / "a" / "lock.json", B=50)["preds"]
    r = ER.excluded_summary(tp, fake_features, "ridge"); t = r["table"]
    assert t.loc["all_test_rows", "rows"] == t.loc["scored_in_headline_table", "rows"] + t.loc["left_out_of_headline_table", "rows"]
    assert np.isclose(r["share_left_out"], t.loc["left_out_of_headline_table", "rows"] / t.loc["all_test_rows", "rows"])
    assert (r["by_date"].rows_left_out.is_monotonic_decreasing) and len(r["by_date"]) <= 10
