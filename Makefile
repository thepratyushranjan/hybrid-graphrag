.PHONY: env up down logs ps reset lock test seed search eval

UV_IMAGE := ghcr.io/astral-sh/uv:python3.11-bookworm-slim

env:            ## create .env from the template if missing
	@test -f .env || cp .env.example .env

up: env         ## build and start qdrant, neo4j, api and ui
	docker compose up -d --build

down:           ## stop everything (keeps data)
	docker compose down

logs:           ## follow api + ui logs
	docker compose logs -f api ui

ps:             ## show container health
	docker compose ps

reset:          ## stop everything and DELETE qdrant + neo4j data
	docker compose down -v

lock:           ## regenerate the fully pinned requirements.txt from requirements.in (inside Docker)
	docker run --rm -u $$(id -u):$$(id -g) -e UV_CACHE_DIR=/tmp/uv -v "$$PWD":/w -w /w $(UV_IMAGE) \
		uv pip compile requirements.in -o requirements.txt --python-version 3.11 \
		--extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
		--emit-index-url --no-header

test: env       ## run the test suite inside the api container (needs `make up`)
	docker compose run --rm --no-deps -v "$$PWD/tests":/app/tests api pytest -q tests

seed:           ## ingest the sample documents in data/samples (needs `make up`)
	docker compose exec api python scripts/seed.py

search:         ## vector search, e.g. make search q="road accident" args="--language hi"
	docker compose exec api python scripts/search.py "$(q)" $(args)

eval:           ## run the Ragas evaluation
	@echo "eval: available once the query pipeline is built (bonus: Ragas)"; exit 1
