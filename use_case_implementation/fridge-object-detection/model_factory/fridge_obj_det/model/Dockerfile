FROM python:3.8-slim

RUN apt-get update &&\
    apt-get install -y --no-install-recommends gcc curl git libssl-dev openssl wget build-essential zlib1g-dev

RUN curl https://pyenv.run | bash && \
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile && \
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile && \
    echo 'eval "$(pyenv init -)"' >> ~/.profile

ENV PATH="/root/.pyenv/bin:${PATH}"
ENV PYENV_ROOT="/root/.pyenv"

RUN pyenv install 3.9.12 &&\
    pyenv global 3.9.12

ENV POETRY_HOME="/root/.poetry"
ENV PATH="/root/.poetry/bin:${PATH}"

RUN curl -sSL https://install.python-poetry.org |  python3 -
RUN poetry config virtualenvs.prefer-active-python true
ENV PATH="/root/.pyenv/versions/3.9.12/bin:${PATH}"

# install packages
COPY pyproject.toml poetry.lock .python-version /app/
WORKDIR /app
RUN poetry config installer.max-workers 10 &&\
    poetry install --no-interaction --no-ansi -vvv

# Setup env for the the container
ENV AZUREML_MODEL_DIR=/app/azureml-models
ENV MODEL_LOG_PATH=/app/logs/
ENV GUNICORN_LOG_LEVEL=debug

# Create dirs
RUN mkdir -p $AZUREML_MODEL_DIR &&\
    mkdir -p $MODEL_LOG_PATH

# Copy scoring files
COPY scoring /app/scoring
COPY server /app/server

# Copy Model
COPY model_artifacts/* /app/azureml-models/

EXPOSE 8080

CMD poetry run gunicorn --timeout 600 -b=0.0.0.0:8080 --capture-output --log-level ${GUNICORN_LOG_LEVEL} server.__main__:app