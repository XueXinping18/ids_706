install:
	pip3 install --upgrade pip &&\
		pip3 install -r requirements.txt

format:
	black *.py

lint:
	flake8 *.py --max-line-length 100

test:
	python -m pytest -vv test_gold_analysis.py

analyze:
	python gold_analysis.py

docker-build:
	docker build -t gold-analysis .

docker-run:
	docker run --rm gold-analysis

clean:
	rm -rf __pycache__ .pytest_cache *.png

all: install format lint test

.PHONY: install format lint test analyze docker-build docker-run clean all
