"""
Centralized logging configuration for the probabilistic forecasting and inventory pipeline.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_file: str | Path | None = None,
    level: str | int = "INFO",
) -> logging.Logger:
    """
    Configures root logging with a consistent timestamped format to stdout and,
    optionally, to a ``.log`` file.

    Parameters
    ----------
    log_file : str or Path, optional
        If provided, all log messages are also written to this file (UTF-8).
        Parent directories are created automatically.
    level : str or int, default "INFO"
        Logging level (e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``).
    """
    numeric_level = (
        getattr(logging, level.upper(), logging.INFO) if isinstance(level, str) else int(level)
    )
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicate messages on repeated calls
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        if log_path.suffix == "":
            log_path = log_path.with_suffix(".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        root_logger.info("Logging to file: %s", log_path.resolve())

    return root_logger

