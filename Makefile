PYTHON ?= python3
export PYTHONDONTWRITEBYTECODE := 1

.PHONY: docs preview check-docs final-docs benchmark pilot score audit release
docs:
	$(PYTHON) submission/build.py

preview:
	$(PYTHON) submission/build.py --preview "$(MODEL)"

check-docs:
	$(PYTHON) submission/build.py --check

final-docs:
	$(PYTHON) submission/build.py --final

benchmark:
	$(PYTHON) benchmark/setup.py

pilot:
	$(PYTHON) benchmark/validate.py "$(MODEL)" --pilot --runs "$(RUNS)"

score:
	$(PYTHON) benchmark/validate.py "$(MODEL)" --export --runs "$(RUNS)"

audit:
	$(PYTHON) benchmark/validate.py "$(MODEL)" --runs "$(RUNS)"

release:
	$(PYTHON) submission/release.py
