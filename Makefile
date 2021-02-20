VENV_DIR ?= .venv
VENV_RUN = . $(VENV_DIR)/bin/activate
PIP_CMD ?= pip
TEST_PATH ?= tests

usage:          ## Show this help
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

install:        ## Install dependencies in local virtualenv
	(test `which virtualenv` || $(PIP_CMD) install --user virtualenv) && \
		(test -e $(VENV_DIR) || virtualenv $(VENV_OPTS) $(VENV_DIR))
	test ! -e requirements.txt || ($(VENV_RUN); $(PIP_CMD) install -r requirements.txt)

test:           ## Run local unit and integration tests
	$(VENV_RUN); PYTHONPATH=`pwd` nosetests $(NOSE_ARGS) --with-timer --with-coverage --logging-level=WARNING --nocapture --no-skip --exe --cover-erase --cover-tests --cover-inclusive --cover-package=cpopservice --with-xunit --exclude='$(VENV_DIR).*' $(TEST_PATH)

start:          ## Start up the CPOP server
	$(VENV_RUN); PYTHONPATH=`pwd` python cpopservice/server.py

lint:           ## Run code linter to check code style
	($(VENV_RUN); flake8 --inline-quotes=single --show-source --max-line-length=120 --ignore=E128,W504 --exclude=$(VENV_DIR)* .)

.PHONY: usage test
