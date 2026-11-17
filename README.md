# Seven-day-ahead store sales forecasting (Rossmann)

ES 678 individual project. Forecasts daily sales for each of 1,115 Rossmann stores seven days ahead, with an 80 percent
prediction interval, using only information known seven days earlier.

## Result (test period 1 April to 31 July 2015, scored once)

| Model | MAE in euros (95% interval) | WAPE |
|---|---|---|
| Same weekday last week | 2,139 (2,092 to 2,185) | 29.7% |
| Mean of last 4 same weekdays | 1,273 (1,247 to 1,300) | 17.7% |
| Linear regression | 971 (949 to 994) | 13.5% |
| Ridge | 787 (768 to 809) | 10.9% |
| Lasso | 781 (762 to 803) | 10.9% |
| **LightGBM (chosen by a rule fixed in advance)** | **601 (588 to 615)** | **8.4%** |

Intervals come from resampling whole stores 1,000 times. Conformal intervals cover 83.8 percent of test days at a nominal
80 percent, but fail on public holidays and Easter. The model is weak for stores with no history. Full discussion: `paper/paper.tex`,
`decision_log.md` and the files in `artifacts/`.

## Data (not included)
Rossmann Store Sales, Kaggle competition: https://www.kaggle.com/c/rossmann-store-sales . Free Kaggle account needed; accept the
competition rules and read what they permit. Download `train.csv` and `store.csv` into `data/raw/` (never commit them).
Kaggle token: save as `~/.kaggle/kaggle.json`, outside this repository.

## Reproduce everything
```
python -m venv .venv && .venv\Scripts\activate         # Windows; use source .venv/bin/activate elsewhere
pip install -r requirements.txt
python -m pytest -q tests                              # 70 tests, no data needed (they use generated data)
python src/audit.py                                    # data audit            -> artifacts/audit_report.json
python src/etl.py --fresh                              # warehouse + 217 reconciliation checks (30 daily loads, then bulk)
python src/features.py                                 # 36 point-in-time features -> data/processed/features_h7.pkl
python src/tune.py                                     # Optuna, 40 trials x 3 seeds per model -> artifacts/best_params.json
python src/run_cv.py --models ridge,lasso,lightgbm --params artifacts/best_params.json --extra-seeds 2,3,4,5 --sanity
python src/select_model.py                             # the pre-declared choice rule -> artifacts/model_choice.json
git tag freeze-before-test                             # commit everything first
python src/final.py                                    # scores the test period ONCE (writes artifacts/test_lock.json)
python src/intervals.py ; python src/failure_audit.py ; python src/excluded_rows.py
python src/ablation.py                                 # feature groups, horizon sweep, Customers leakage demonstration
python src/make_figures.py                             # figures -> paper/figures
```
All split dates, seeds and budgets are in `config.yaml`, which was committed before any model was compared.
`final.py` refuses a second run; the test period is not used for tuning, feature choice or model choice.

## Use the model
```
uvicorn serve:app --app-dir src --port 8000            # API docs at http://127.0.0.1:8000/docs
streamlit run src/app.py                               # web page with interval, history and feature effects
```
`POST /forecast` takes `store_id`, `target_date` and optional `promo`, `state_holiday`, `school_holiday` overrides; it returns the
forecast, the 80 percent interval, the forecast origin (target minus 7 days), the model version and the actual sales.
It runs in replay mode on the loaded period. Forecasts beyond the loaded data would need a calendar extension (not built).
Overrides show how the model reacts; they are not the causal effect of a promotion.

## Layout
`sql/` star schema | `src/` etl, features, tuning, evaluation, intervals, audits, API | `tests/` 70 tests | `artifacts/` results and the
saved model | `paper/` LaTeX source and figures | `decision_log.md` every design decision with its reason.

## Limits
One retailer, one country, 2013 to 2015. Promotions were not randomised, so feature effects are associations. Christmas is
absent from the test period and Easter appears once. Sales understate demand when stock runs out. 5.5 percent of test rows have no
last-week value and are left out of the headline scores (the model does better on them).

## AI-use disclosure and citations

AI-use disclosure: Claude (Anthropic) helped me plan the project and write the code. The dataset, the data cleaning, running the code and the evaluation were done by me on my own machine, and every number in the paper comes from my own outputs.

Citations: dataset Rossmann Store Sales, Kaggle, 2015. Software: pandas (McKinney 2010), NumPy (Harris et al. 2020), SciPy (Virtanen et al. 2020), scikit-learn (Pedregosa et al. 2011), LightGBM (Ke et al. 2017), Optuna (Akiba et al. 2019), Matplotlib (Hunter 2007), SQLite, and FastAPI and Streamlit for the API and demo page. Methods: Hyndman and Koehler 2006, Angelopoulos and Bates 2021, Duan 1983.

The artifacts folder holds derived result tables and the trained model; audit_largest_errors.csv and audit_worst_stores.csv list a few stores and days from the competition data to document the error analysis. The full dataset is not included.
