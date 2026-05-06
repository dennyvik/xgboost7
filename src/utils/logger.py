import logging
from pathlib import Path
from typing import Any


def setup_logger(run_dir: str | Path, level: int = logging.INFO) -> logging.Logger:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(level)

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(run_path / "logs.txt", encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def log_kv(logger: logging.Logger, message: str, **kwargs: Any) -> None:
    key_values = " | ".join(f"{key}={value}" for key, value in kwargs.items())
    logger.info(f"{message} | {key_values}" if key_values else message)