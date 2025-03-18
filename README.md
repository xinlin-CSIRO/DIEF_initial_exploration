# dief-competition-2

Project template suitable for most python projects using the `uv` toolchain with basic tooling pre-installed.
It is designed to have a reasonable balance between eas of use and good software practices, and contains many features
which are there if you want them, but can be ignored until you do.

This template attempts to providing tooling based on the [Coding Best Practices](https://confluence.csiro.au/x/xeQRd)
confluence page.
It is largely based on the older [python-poetry-template](https://github.com/csiro-energy-systems/_PythonTemplate), but
update to use `uv` instead of `poetry`.

[![Unit Tests](https://github.com/csiro-energy-systems/dief-competition-2/actions/workflows/ci.yml/badge.svg)](https://github.com/csiro-energy-systems/dief-competition-2/actions/workflows/ci.yml)
[![Docker Unit Tests](https://github.com/csiro-energy-systems/dief-competition-2/actions/workflows/docker-image.yml/badge.svg)](https://github.com/csiro-energy-systems/dief-competition-2/actions/workflows/docker-image.yml)
[![Release](https://github.com/csiro-energy-systems/dief-competition-2/actions/workflows/release.yml/badge.svg)](https://github.com/csiro-energy-systems/dief-competition-2/actions/workflows/release.yml)

## Quick start:

### Prerequisites

If you don't already have it, just install `uv`:

#### Install uv

To install uv, see the instructions: https://docs.astral.sh/uv/getting-started/installation/, but in short:

In bash on (most) Linux systems and Mac:

```bash
curl -LsSf https://astral.sh/uv/0.4.6/install.sh | sh
# restart your shell and make sure `uv --version` works
```

In Windows, from any shell run:

```shell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" # windows
# restart your shell and make sure `uv --version` works
```

### Using the template

If cloning this repo, and the tools above are installed, run these commands to set up the environment for development.

:warning: Please use `bash` (mac/linux/wsl) or `git-bash` (windows) for the following commands:

```bash
# Set these environment variables to the desired values for your new project
export NEW_PROJECT_NAME="your_new_project_name_here"
export NEW_PROJECT_AUTHOR="Your Name"
export NEW_PROJECT_EMAIL="your.name@csiro.au"

# Select project python>=3.12 version to use/download. See `uv python list --all-versions` for available versions
export NEW_PROJECT_PYTHON="3.13"

# Clone the repo
git clone git@github.com:csiro-energy-systems/dief-competition-2.git $NEW_PROJECT_NAME; cd $NEW_PROJECT_NAME

# Templatise (initialise) the project.  This renames everything using the env vars above, sets up the venv
# and untracks/inits the git repo for you.
uv run --python=$NEW_PROJECT_PYTHON poe rename_template --execute --untrack
```

If everything worked, at this point you will have a new templated project, ready to push to a new git repo and begin
coding in.

There are a few other optional steps you might want to take now:

```bash
# (Optional) run tests locally to make sure things are working.
uv run poe test

# if you don't want to use Docker, you can remove the docker-related files:
rm docker-compose.yml Dockerfile workflows/docker-image.yml

# if you want to use Docker with our Azure pypi index, update your .env.secret file:
# NOTE: Make sure you replace YOUR_PAT_HERE with your token from created at https://dev.azure.com/csiro-energy/_usersSettings/tokens
UV_EXTRA_INDEX_URL=https://dummy:YOUR_PAT_HERE@pkgs.dev.azure.com/csiro-energy/csiro-energy/_packaging/csiro-python-packages/pypi/simple/

# (Optional) Install git-lfs to store larger files in the repo without bloating the history
git lfs install # windows
apt-get install git-lfs # linux
```

Other useful commands include:

:warning: Again, please use `bash` (mac/linux/wsl) or `git-bash` (windows)

```bash
# list all built-in build tasks
uv run poe --help

# run pre-commit linters and formatters
uv run poe lint

# generate and show html doc
uv run poe doc
uv run poe show_doc

# Add private pypi index (Azure Devops in this example) for publishing release packages wheels to, and installing packages from.
# NOTE: Make sure you replace YOUR_PAT_HERE with your token from created in azure devops, e.g. https://dev.azure.com/csiro-energy/_usersSettings/tokens
# See https://confluence.csiro.au/display/GEES/Poetry+Cheat+Sheet#PoetryCheatSheet-UV for more details
export UV_EXTRA_INDEX_URL=https://dummy:YOUR_PAT_HERE@pkgs.dev.azure.com/csiro-energy/csiro-energy/_packaging/csiro-python-packages/pypi/simple/

# build a redistributable python package
uv build

# publish a built wheel to the private csiroenergy pypi repo. Only works once for each unique project version (in pyproject.toml).
twine upload dist/*

# build a docker image, and define two containers, one for tests, and one for launching your code
docker compose build

# run tests in docker container
docker compose run test

# check current project version
uv run poe version

# Publish wheel to package repo.  Set $PUBLISH_URL, $PUBLISH_USER and $PUBLISH_PASSWORD first.
uv run poe publish

# To trigger a github release, increment the pyproject version create and push a matching git tag.
# This creates a wheel and docker image, and uploads them to the pypi repo and github container registry, and adds to
# the Release page on github.
uv run poe release

# Run the published wheel in a temporary sandboxed virtual environment.  Only requires `uv`, not a cloned repo or even python.  Run `poe publish` first.
uvx $NEW_PROJECT_NAME
```

## Features

The main features and packages provided by this template are:

| Feature                | Tool                 | Description                                                                                                          |
|------------------------|----------------------|----------------------------------------------------------------------------------------------------------------------|
| Environment Manager    | uv                   | fast, robust, self-contained, package/build manager (`uv --help`)                                                    |
| Build Tasks            | poethepoet           | cross-platform dev task runner (`poe` command, and see `pyproject.toml` [tool.poe.tasks] section).                   |
| Logging                | loguru               | python logging made easy (`from loguru import logger`)                                                               |
| Unit Testing           | pytest               | unit testing & coverage (`poe test`)                                                                                 |
| Template Installation  | custom python script | see [Using the template](#Using-the-template) section above for details (`uv run poe rename_template --help`)        |
| Continuous Integration | github actions       | pre-configured for Continuous integration/unit testing after each push (`.github/workflows/*.yml`)                   |
| Doc Generation         | sphinx               | markdown to html doc generator (`poe doc`)                                                                           |
| Licensing              | pip-licenses         | exports a list of libraries & licenses, adds to doc (`uv run pip-licenses`)                                          |
| Code Quality           | pre-commit           | pre-configured with auto-formatting (ruff-format), linting (ruff), optional static type checking (mypy), etc.        |
| Secret Scanning        | gitleaks             | scans for secrets before commits (`pre-commit run gitleaks -a`)                                                      |
| Code Security          | bandit               | checks code for common security flaws (`pre-commit run bandit -a`).                                                  |
| Containerisation       | docker               | minimal configuration `Dockerfile` included  (`poe docker_build`)                                                    |
| Data Tracking          | git-lfs              | git Large File System (LFS) (`lfs track data/**/*`)                                                                  |
| Releases               | github, azure devops | create a release on github, and publish a package wheel (`poe release`)                                              |
| Runtime Env Management | python-dotenv        | manage per-user runtime environment variables via `.env` (per-user variables) and `.env.secret` (secret credentials) |
| Git Ignore template    | gitignore template   | sensible `.gitignore` template for python projects from [github](https://github.com/github/gitignore)                |

## Developer Setup

Ignore this section if you're not maintaining this template repo.

To set up a new development environment from an existing repo in git:

```bash
git clone git+ssh://this.repo.url
cd this.repo.name
uv sync
uv run poe test
```
