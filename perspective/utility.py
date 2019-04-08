"""Utility functions applicable to many things."""

import logging
import os.path
import sys

def check_output_necessary(output_path, overwrite):
    """Determine whether a step is necessary by checking for its existence/overwrite combo.

    Returns true if should continue with step, false if can skip.
    """

    logging.debug("Checking for existence of '%s'...", output_path)

    if os.path.isfile(output_path):
        logging.debug("Output found.")

        # check if should overwite the existing output or not
        if overwrite:
            logging.debug("Overwrite requested, continuing...")
            logging.warning("Overwriting an existing output '%s'!", output_path)
            return True

        logging.debug("No overwrite requested, skip step...")
        return False

    # if this point hit, the file doesn't exist yet
    return True

def init_logging(log_path):
    """Sets up logging config, including associated file output."""
    log_formatter = logging.Formatter("%(asctime)s - %(filename)s - %(levelname)s - %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
