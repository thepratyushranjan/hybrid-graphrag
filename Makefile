.PHONY: env up down logs ps reset

env:            ## create .env from the template if missing
	@test -f .env || cp .env.example .env

up: env         ## build and start qdrant, neo4j and the api
	docker compose up -d --build

down:           ## stop everything (keeps data)
	docker compose down

logs:           ## follow api logs
	docker compose logs -f api

ps:             ## show container health
	docker compose ps

reset:          ## stop everything and DELETE qdrant + neo4j data
	docker compose down -v
