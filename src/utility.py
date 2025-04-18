import logging
from datetime import datetime
import os

from rich.logging import RichHandler
from rich.console import Console

console = Console()

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


def load_data():
    file_path = "data/Food_Inspections_20250216.csv"
    df = pd.read_csv(file_path, header=[0])
    console.log(f"[bold green]SUCCESS[/bold green] File loaded from {file_path}")
    console.log(df.head())

    return df


def save_data(df):
    file_path = "data/Food_Inspections_20250216_preprocessed.parquet"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save as Parquet
    df.to_parquet(file_path, index=False)

    console.log(f"[bold green]SUCCESS[/bold green] File persisted to {file_path}")
