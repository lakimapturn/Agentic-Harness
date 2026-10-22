.PHONY: init install start attach stop bash-example py-example redis-up redis-down

init:
	@echo "Initializing git submodules..."
	@if [ -d .git ]; then git submodule update --init --recursive; else echo "No .git directory (ZIP checkout); PyPI mini-swe-agent will be used."; fi

install:
	@echo "Installing dependencies..."
	uv sync --extra mini --extra test
	@echo "Dependencies installed."
	@echo "Activate the environment with: source .venv/bin/activate"

start:
	@bash scripts/start.sh start

attach:
	@bash scripts/start.sh attach

stop:
	@bash scripts/start.sh stop

bash-example:
	bash examples/run_bash_example.sh

py-example:
	bash examples/run_python_example.sh

redis-up:
	docker compose up -d redis

redis-down:
	docker compose down
