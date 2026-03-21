"""
logger.py
---------
Centralised logging setup for NSL-RAG.
Every module gets its logger from here — never configure logging elsewhere.

Usage:
    from nsl_rag.core.logger import get_logger
    log = get_logger(__name__)
    log.info("Something happened: %s", value)
"""

import logging
import sys
from pathlib import Path


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_LEVEL = logging.DEBUG
DEFAULT_FORMAT = "console"
LOG_FILE_PATH = Path("logs/nsl_rag.log")


# ── Formatters ────────────────────────────────────────────────────────────────


def get_console_formatter() -> logging.Formatter:
    """
    Human-readable formatter for development.
    Example output:
        14:23:01 | DEBUG    | nsl_rag.lattice.node | Node created: payment_service
    """
    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%H:%M:%S"
    )


def get_json_formatter() -> logging.Formatter:
    """
    Structured formatter for production.
    Produces consistent log lines that log aggregators can parse.
    Example output:
        {"time": "14:23:01", "level": "DEBUG", "logger": "nsl_rag.lattice.node", "msg": "..."}
    """
    return logging.Formatter(
        fmt='{"time": "%(asctime)s", "level": "%(levelname)s", '
        '"logger": "%(name)s", "msg": "%(message)s"}',
        datefmt="%H:%M:%S",
    )


# ── Handlers ──────────────────────────────────────────────────────────────────


def get_console_handler(formatter: logging.Formatter) -> logging.StreamHandler:
    """
    Returns a handler that writes to stdout.
    Used in both development and production.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    return handler


def get_file_handler(
    formatter: logging.Formatter, path: Path = LOG_FILE_PATH
) -> logging.FileHandler:
    """
    Returns a handler that writes to a log file.
    Creates the logs/ directory if it does not exist.
    Used in production only.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(formatter)
    return handler


# ── Core Setup ────────────────────────────────────────────────────────────────


def setup_logging(
    level: str = "DEBUG",
    fmt: str = "console",
    file_enabled: bool = False,
    file_path: Path = LOG_FILE_PATH,
) -> None:
    """
    Configure the root nsl_rag logger.
    Call this ONCE at application startup — in main.py or the pipeline entry point.
    All child loggers (nsl_rag.lattice, nsl_rag.retrieval etc.) inherit this config.

    Args:
        level:        Log level string — DEBUG, INFO, WARNING, ERROR.
        fmt:          Format style — console or json.
        file_enabled: Whether to also write logs to a file.
        file_path:    Path to the log file if file_enabled is True.
    """
    root_logger = logging.getLogger("nsl_rag")

    # Avoid adding duplicate handlers on repeated calls
    if root_logger.handlers:
        return

    # Resolve level
    numeric_level = getattr(logging, level.upper(), logging.DEBUG)
    root_logger.setLevel(numeric_level)

    # Select formatter
    formatter = get_console_formatter() if fmt == "console" else get_json_formatter()

    # Always add console handler
    root_logger.addHandler(get_console_handler(formatter))

    # Optionally add file handler
    if file_enabled:
        root_logger.addHandler(get_file_handler(formatter, file_path))

    # Prevent logs from bubbling up to the Python root logger
    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger for a module.
    Always pass __name__ as the argument.

    Example:
        log = get_logger(__name__)

    This produces loggers named:
        nsl_rag.lattice.node
        nsl_rag.retrieval.intent
        nsl_rag.reasoning.judge
    All inherit config from the nsl_rag root logger.
    """
    return logging.getLogger(name)
