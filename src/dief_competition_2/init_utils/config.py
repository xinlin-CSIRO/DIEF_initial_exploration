# The Software is copyright (c) Commonwealth Scientific and Industrial Research Organisation (CSIRO) 2023-2024.
# see https://loguru.readthedocs.io/en/stable/api/type_hints.html#module-autodoc_stub_file.loguru
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import loguru
from loguru import logger

# Get entry point file name as default log name
default_log_name = Path(sys.argv[0]).stem
default_log_name = "log" if default_log_name == "" else default_log_name


def get_logger(
    log_name: str = default_log_name,
    log_dir: str = "logs",
    level: str | None = None,
    file_mode: str = "a+",
    diagnose: bool = False,
) -> loguru.Logger:
    """Return a configured loguru logger.

    Call this once from entrypoints to set up a new logger.
    In non-entrypoint modules, just use `from loguru import logger` directly.

    To set the log level, use the `LOGURU_LEVEL` environment variable before or during runtime. E.g. `os.environ["LOGURU_LEVEL"] = "INFO"`
    Available levels are `TRACE`, `DEBUG`, `INFO`, `SUCCESS`, `WARNING`, `ERROR`, and `CRITICAL`. Default is `INFO`.

    Log file will be written to `f"{log_dir}/{log_name}.log"`

    See https://github.com/Delgan/loguru#suitable-for-scripts-and-libraries
    From loguru import Record, RecordFile # See these classes for all the available format strings

    :param default_log_name: Name of the log. Corresponding log file will be called {log_name}.log in the .
    :param log_dir: Directory to write the log file to. Default is the current working directory.
    :param level: Log level. Default is `DEBUG`.
    :param file_mode: File mode to open the log file in. Default is `a+`.
    :param diagnose: Whether to include diagnostic information in the log. Default is `False`.

    Returns:
        Logger: A configured loguru logger.
    """
    # set global log level via env var.  Set to INFO if not already set.
    if os.getenv("LOGURU_LEVEL") is None:
        os.environ["LOGURU_LEVEL"] = "DEBUG"
    if level is None:
        level = os.getenv("LOGURU_LEVEL")

    format_str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> : <level>{message}</level> "
        '(<cyan>{name}:{thread.name}:pid-{process}</cyan> "<cyan>{file.path}</cyan>:<cyan>{line}</cyan>")'
    )
    log_config: dict[Any, Any] = {
        "handlers": [
            {
                "sink": sys.stdout,
                "diagnose": diagnose,
                "format": format_str,
                "level": level,
            },
            {
                "sink": f"{log_dir if log_dir is not None else 'output'}/{log_name}.log",
                "mode": file_mode,
                "colorize": False,
                "serialize": False,
                "diagnose": diagnose,
                "rotation": "10 MB",
                "compression": "zip",
                "format": format_str,
                "level": level,
            },
        ],
    }
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger.configure(**log_config)

    return logger
