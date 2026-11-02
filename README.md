# Seven-day-ahead store sales forecasting (Rossmann)

ES 678 individual project. Forecast daily sales per store 7 days ahead with prediction intervals.

## Start here (Phase 1 and 2)
1. `python -m venv .venv` then activate it, then `pip install -r requirements.txt`.
2. Make a free Kaggle account, open the Rossmann Store Sales competition page, join it and accept the rules. Read the rules page for permitted use.
3. Create an API token in your Kaggle account settings and save it as `~/.kaggle/kaggle.json` (never inside this repository).
4. `kaggle competitions download -c rossmann-store-sales -p data/raw` and unzip train.csv and store.csv into `data/raw/`.
5. `python src/audit.py` and write one line per finding in `decision_log.md`.
6. Commit `config.yaml` BEFORE comparing any model.

No Kaggle yet? Run `make fake` to test the scripts on generated data.
Raw data, the warehouse file and credentials are never committed (see `.gitignore`).
