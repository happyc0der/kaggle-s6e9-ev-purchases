.PHONY: data test base fe
PY=.venv/bin/python
data:
	.venv/bin/kaggle competitions download -c playground-series-s6e9 -p data/raw && cd data/raw && unzip -o playground-series-s6e9.zip && rm playground-series-s6e9.zip
	.venv/bin/kaggle datasets download itzzomkar/ev-adoption-behavior-and-range-anxiety -p data/raw && cd data/raw && unzip -o ev-adoption-behavior-and-range-anxiety.zip && rm ev-adoption-behavior-and-range-anxiety.zip
test:
	$(PY) -m pytest -q tests
base:
	$(PY) scripts_phase0.py base
fe:
	$(PY) scripts_phase0.py fe
