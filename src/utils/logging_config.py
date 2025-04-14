# src/utils/logging_config.py
"""
Utility function for basic logging setup.
"""

import logging
import sys
import os
from datetime import datetime

def setup_logging(
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_dir: str = 'logs',
    log_filename_prefix: str = 'experiment'
):
    """
    Sets up basic logging configuration.

    Logs to both console (stdout) and optionally to a file.

    Args:
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        log_to_file (bool): If True, logs to a timestamped file in log_dir.
        log_dir (str): Directory to store log files.
        log_filename_prefix (str): Prefix for the log file name.
    """
    log_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)-5.5s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to avoid duplicate logging
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    # File Handler (Optional)
    if log_to_file:
        try:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"{log_filename_prefix}_{timestamp}.log")

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(log_formatter)
            root_logger.addHandler(file_handler)
            logging.info(f"Logging setup complete. Also logging to file: {log_file}")
        except Exception as e:
            logging.error(f"Failed to set up file logging to {log_dir}: {e}", exc_info=True)
    else:
        logging.info("Logging setup complete (console only).")

# Example usage (typically called once at the start of a main script)
# if __name__ == "__main__":
#     setup_logging(level=logging.DEBUG)
#     logging.debug("This is a debug message.")
#     logging.info("This is an info message.")
#     logging.warning("This is a warning message.")
#     logging.error("This is an error message.")
