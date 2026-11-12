audit:
	python src/audit.py
fake:
	python tests/make_fake_data.py && python src/audit.py data/fake
warehouse:
	python src/etl.py --fresh
test:
	python -m pytest -q tests
features:
	python src/features.py
cv:
	python src/run_cv.py --sanity
diagnose:
	python src/diagnose_cv.py
tune:
	python src/tune.py
select:
	python src/select_model.py
final:
	python src/final.py
intervals:
	python src/intervals.py
ablation:
	python src/ablation.py
failure:
	python src/failure_audit.py
figures:
	python src/make_figures.py
serve:
	uvicorn serve:app --app-dir src --port 8000
ui:
	streamlit run src/app.py
paper:
	cd paper && pdflatex paper.tex && pdflatex paper.tex
promo-cycle:
	python src/check_promo_cycle.py
