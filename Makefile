# Semantic Anarchy — convenience targets. `make` on its own lists them.
#
# Env overrides pass straight through, e.g.:
#   make run SA_PORT=9000
#   make run SA_HOST=0.0.0.0

REPO := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
PY   := $(REPO)/.venv/bin/python
FE   := $(REPO)/webui/frontend

.DEFAULT_GOAL := help
.PHONY: help run start dev build install restart stop logs test

help:  ## show this list
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
	  | sort | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

run:  ## start the dashboard (foreground, Ctrl-C to stop)
	@$(REPO)/webui/run.sh

start: run  ## alias for `make run`

dev:  ## Vite dev server on :5173 (run `make run` in another shell first)
	@cd $(FE) && npm run dev

build:  ## rebuild the frontend into webui/frontend/dist
	@cd $(FE) && npm run build

install:  ## install frontend npm deps
	@cd $(FE) && npm install

restart:  ## detached restart; refuses while a GPU job runs (make restart FORCE=--force)
	@SA_PYTHON=$${SA_PYTHON:-$(PY)} $(REPO)/webui/restart.sh $(FORCE)

stop:  ## kill a detached dashboard
	@pkill -f "webui/app.py" && echo "stopped" || echo "not running"

logs:  ## tail the detached dashboard log
	@tail -f /tmp/sa_webui.log

test:  ## run the (torch-free) test suite
	@$(PY) -m pytest -q
