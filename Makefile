
checks: env
	env/bin/mypy *.py --ignore-missing-imports

env:
	python3 -m venv env
	source env/bin/activate && pip3 install --no-cache-dir -r requirements.txt
	echo
	echo remember to source env/bin/activate
	echo

container:
	cd redis && docker build -t jlewallen/feevee-redis .
	docker build -t jlewallen/feevee .
