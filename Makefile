install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

lint:
	flake8 hello.py gold_analysis.py

test:
	python -m pytest -vv --cov=hello test_hello.py

# New target for running the data analysis
analyze:
	python gold_analysis.py

clean:
	rm -rf **pycache** .pytest_cache .coverage *.png

all: install format lint test
