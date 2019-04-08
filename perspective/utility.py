"""Utility functions applicable to many things."""

import argparse
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
        logging.info("Cached version found.")

        # check if should overwite the existing output or not
        if overwrite:
            logging.debug("Overwrite requested, continuing...")
            logging.warning("Overwriting an existing output '%s'!", output_path)
            return True

        logging.debug("No overwrite requested, skip step...")
        return False

    # if this point hit, the file doesn't exist yet
    return True

def init_logging(log_path=None):
    """Sets up logging config, including associated file output."""
    log_formatter = logging.Formatter("%(asctime)s - %(filename)s - %(levelname)s - %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if log_path is not None:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

def add_common_parsing(parser):
    """Add commonly used options to argument parser."""
    parser.add_argument(
        "-l",
        "--log",
        dest="log_path",
        type=str,
        required=False,
        metavar="<str>",
        help="The path to the log file append to",
    )
    parser.add_argument(
        "-d",
        "--experiment-dir",
        dest="experiment_path",
        type=str,
        required=True,
        metavar="<str>",
        help="The path to the experiment directory",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_path",
        type=str,
        required=False,
        metavar="<str>",
        help="Input path relative to experiment path (or absolute path)",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        type=str,
        required=False,
        metavar="<str>",
        help="Output path relative to experiment path",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Specify this flag to overwrite existing output data if they exist",
    )
    parser.add_argument(
        "-c",
        "--count",
        dest="count",
        type=int,
        required=False,
        default=-1,
        metavar="<int>",
        help="The number of items to use",
    )
    parser.add_argument(
        "-w",
        "--workers",
        dest="workers",
        type=int,
        required=False,
        default=1,
        metavar="<int>",
        help="The number of worker threads to use",
    )
    return parser

def fix_paths(experiment_path, input_path, output_path):
    """Resolve any relative paths that are passed."""
    if not os.path.isabs(input_path):
        input_path = os.path.abspath(experiment_path + "/" + input_path)
    output_path = os.path.abspath(experiment_path + "/" + output_path)

    return input_path, output_path
