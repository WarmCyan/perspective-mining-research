"""Utility functions applicable to many things."""

import logging
import os.path

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
