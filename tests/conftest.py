import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pytest
from make_fake_data import make
import etl, features
from utils import load_config

@pytest.fixture(scope="session")
def fake_features(tmp_path_factory):
    """Features built from a small fake warehouse, shared by the modelling tests."""
    d = tmp_path_factory.mktemp("model")
    raw = make(d / "raw", n_stores=40)
    conn = etl.connect(d / "m.db")
    etl.build(conn, raw, replay_days=3, expected_stores=40)
    return features.build_features(conn, load_config(), 7)
