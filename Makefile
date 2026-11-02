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
