poetry run pip install --upgrade setuptools # Ensure setuptools is up to date
poetry lock
poetry install
poetry run pytest tests/
poetry run python src/limit_price.py
poetry run python src/demand_load.py
poetry run python src/auction_optimization.py
