audit:
	python src/audit.py
fake:
	python tests/make_fake_data.py && python src/audit.py data/fake
