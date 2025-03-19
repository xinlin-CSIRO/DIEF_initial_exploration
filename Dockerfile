# Docker file for hosting dief_competition_2. Builds your project and its dependencies into a docker container.
# This is mostly just the filesystem setup. Use docker compose for runtime stuff (eg volumes, actual entrypoints etc)
# TODO update to multistage build, eg. https://github.com/astral-sh/uv-docker-example/blob/main/multistage.Dockerfile

ARG PYTHON_VERSION=3.12

FROM ghcr.io/astral-sh/uv:python$PYTHON_VERSION-bookworm-slim

# Arguments defined before a FROM will be unavailable after the FROM unless we re-specify the arg
ARG PROJECT_NAME=dief_competition_2
ARG UV_VERSION=0.4.22

WORKDIR /usr/src/app/$PROJECT_NAME/

# install uv
RUN apt-get update && apt-get install curl git -y && apt-get clean autoclean && apt-get autoremove --yes

# Pre-compile python bytecode for faster startup
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# set up python environment
COPY uv.lock pyproject.toml ./

# mount built-time secrets from .env.secret (e.g. private package repo credentials, like UV_EXTRA_INDEX_URL)
# then export them to environment variables, which only last for the duration of the RUN command and don't get stored in the layer.
RUN --mount=type=secret,id=build_secret,required \
    --mount=type=cache,target=/root/.cache/uv \
    export $(grep -v "^#" /run/secrets/build_secret | xargs) && \
    uv sync --frozen --no-install-project

## Now copy our own source and "install" it (adds to pythonpath) - we do this as a separate step to avoid reinstalling all dependencies if only our source changes
## Note: make sure anything to skip is added to .dockerignore
COPY src/ src/
COPY data/ data/
COPY tests/ tests/
COPY scripts/ scripts/
COPY . .
COPY README.md README.md
RUN --mount=type=secret,id=build_secret,required \
    --mount=type=cache,target=/root/.cache/uv \
    export $(grep -v "^#" /run/secrets/build_secret | xargs) && \
    uv sync --frozen

# Clean up UV cache to reduce image size
RUN uv cache clean

# just launch bash for debugging. Specify real entrypoints in docker compose.yml
ENTRYPOINT ["/bin/bash"]
