FROM python:3.10
ADD poetry.lock pyproject.toml ./
RUN pip install poetry
RUN poetry install --no-root
ADD main.py .
CMD poetry run python main.py