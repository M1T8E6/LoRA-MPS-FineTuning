prepare-env:
	./scripts/prepare-env.sh

install:
	pip install -r requirements.txt

uv-install:
	uv pip install -r requirements.txt

login:
	huggingface-cli login

run:
	streamlit run app.py --server.port 8501 --server.headless true

lint:
	pylint app --fail-under=9.0 --fail-on=E
