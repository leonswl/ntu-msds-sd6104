import logging
from datetime import datetime
import os

from rich.logging import RichHandler

def setup_logger(subdir: str):
    log_dir = os.path.join("log", subdir)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

    logger = logging.getLogger(subdir)
    logger.setLevel(logging.DEBUG)  # Capture all logs at DEBUG level

    # File handler (logs everything)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # Capture ALL logs
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)

    # # Rich console handler (exclude DEBUG logs)
    rich_handler = RichHandler(show_time=False, show_path=False, rich_tracebacks=True)
    rich_handler.setLevel(logging.INFO)  # Only show INFO and above in terminal

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(rich_handler)

    return logger