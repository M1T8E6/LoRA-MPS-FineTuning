install:
	pip install -r requirements.txt

uv-install:
	uv pip install -r requirements.txt

login:
	huggingface-cli login