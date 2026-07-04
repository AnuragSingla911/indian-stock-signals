.PHONY: help setup setup-ml setup-backend setup-frontend pipeline pipeline-offline evaluate \
        api web test test-ml test-backend test-frontend lint build build-pages docker \
        docker-down clean

help:
	@echo "Indian Stock Signals - make targets:"
	@echo "  setup            Install ml, backend and frontend deps"
	@echo "  pipeline         Run ML pipeline (live data) -> predictions.json"
	@echo "  pipeline-offline Run ML pipeline offline (deterministic sample data)"
	@echo "  evaluate         Walk-forward out-of-sample model metrics"
	@echo "  api              Start FastAPI backend on :8000"
	@echo "  web              Start React dev server on :3000"
	@echo "  test             Run all tests (ml + backend + frontend)"
	@echo "  lint             Lint ml + backend + frontend"
	@echo "  build            Build frontend production bundle"
	@echo "  build-pages      Build static frontend for GitHub Pages"
	@echo "  docker           docker compose up --build"

setup: setup-ml setup-backend setup-frontend

setup-ml:
	cd ml && python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"

setup-backend:
	cd backend && python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"

setup-frontend:
	cd frontend && npm install

pipeline:
	cd ml && . .venv/bin/activate && iss-pipeline -v

pipeline-offline:
	cd ml && . .venv/bin/activate && ISS_OFFLINE=1 iss-pipeline -v

evaluate:
	cd ml && . .venv/bin/activate && iss-evaluate -v

api:
	cd backend && . .venv/bin/activate && uvicorn app.main:app --reload --port 8000

web:
	cd frontend && npm run dev

test: test-ml test-backend test-frontend

test-ml:
	cd ml && . .venv/bin/activate && pytest -q

test-backend:
	cd backend && . .venv/bin/activate && pytest -q

test-frontend:
	cd frontend && npm test

lint:
	cd ml && . .venv/bin/activate && ruff check src tests && mypy src
	cd backend && . .venv/bin/activate && ruff check app tests && mypy app
	cd frontend && npm run lint

build:
	cd frontend && npm run build

build-pages:
	cd frontend && npm run build:pages

docker:
	docker compose up --build

docker-down:
	docker compose down

clean:
	rm -rf ml/.venv backend/.venv frontend/node_modules frontend/dist ml/data/price_cache
