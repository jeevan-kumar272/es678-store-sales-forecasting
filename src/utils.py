import pathlib, yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]

def load_config(path=None):
    with open(path or ROOT / "config.yaml") as f:
        return yaml.safe_load(f)
