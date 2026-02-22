from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "clincot",
    level: int = logging.INFO,
    log_file: Optional[str | Path] = None,
    fmt: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "clincot") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger = setup_logger(name=name)
    return logger
