.venv:  # creates .venv folder if does not exist
	python3.10 -m venv .venv

.venv/bin/uv: .venv # installs latest pip
	.venv/bin/pip install -U uv


install: .venv/bin/uv
	.venv/bin/python3 -m uv pip install -r requirements.txt --index-strategy unsafe-best-match