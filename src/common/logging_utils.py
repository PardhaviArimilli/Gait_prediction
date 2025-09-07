import logging
import os
from logging.handlers import RotatingFileHandler


def get_logger(name: str, log_dir: str = "logs", level: str = "INFO") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, os.getenv("LOG_LEVEL", level).upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = RotatingFileHandler(os.path.join(log_dir, f"{name}.log"), maxBytes=10 * 1024 * 1024, backupCount=10)
    fh.setFormatter(fmt)
    fh.setLevel(logger.level)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logger.level)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.propagate = False
    return logger


def close_logger(logger: logging.Logger) -> None:
    for h in list(logger.handlers):
        try:
            h.flush()
        except Exception:
            pass
        try:
            h.close()
        except Exception:
            pass
        logger.removeHandler(h)
    logger.handlers.clear()
