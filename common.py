#!/usr/bin/python

import logging
import os
import subprocess
import sys
from csv import reader

import bcolors
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

# input coloumns
INPUT_COL0_TS = "timestamps"
INPUT_COL1_MI = "Motion Index"
INPUT_COL2_FREEZE = "Freeze"

# Timeshift header in the input
TIMESHIFT_HEADER = "timeshift"
TIMESHIFT_HEADER_ALT = "shift"

# Number of initial rows to skip.
NUM_INITIAL_ROWS_TO_SKIP = 3
OUTPUT_DIR_NAME = "_output"
CSV_EXT = ".csv"

# Colors
LIGHT_RED_COLOR = "#FFCCCB"
LIGHT_YELLOW_COLOR = "#FFFFED"

# globals
logger = None
input_dir = ""


def convert_obj_to_nan(val: object) -> object:
    """
    Parameters
    ----------
    val - Object to check if it can be converted to a float.

    Returns
    ------
    val -> if `val` can be converted to float; np.nan otherwise.
    """

    if str_is_float(val.strip()):
        return val

    return np.nan


def parse_input_file_into_df(input_file: str, skip_num_initial_rows: int) -> (bool, pd.DataFrame):
    """
    Parse the input file
    Parameters
    ----------
    input_file - Full path to the input file to pars.e

    skip_num_initial_rows - Number of initial rows to skip.

    Returns
    ------
    (bool, pandas.DataFrame)
        bool - True if parsing was successful; False otherwise
        DataFrame - Parsed DataFrame`
    """
    in_col_names = [INPUT_COL0_TS, INPUT_COL1_MI, INPUT_COL2_FREEZE]
    try:
        df = pd.read_csv(input_file, names=in_col_names,
                         dtype={INPUT_COL0_TS: 'float64',
                                INPUT_COL1_MI: 'float64',
                                INPUT_COL2_FREEZE: 'int'},
                         skiprows=skip_num_initial_rows)
    except ValueError:
        logger.warning(
            "Input file(%s) contains invalid (NaN - Not A Number) values. Skipping rows with any NaN.", input_file)
        df = pd.read_csv(input_file, names=in_col_names,
                         skiprows=skip_num_initial_rows,
                         converters={INPUT_COL1_MI: convert_obj_to_nan})

        # Drop the NaN
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Try to convert to the required type again and if this one fails, just skip the file
        # with a warning.
        try:
            df = df.astype({INPUT_COL0_TS: 'float64',
                            INPUT_COL1_MI: 'float64',
                            INPUT_COL2_FREEZE: 'int'})
        except ValueError:
            logger.warning("Invalid values in the input file(%s)", input_file)
            return False, None

    if df.isnull().values.any():
        logger.warning(
            "Input file(%s) contains invalid (NaN - Not A Number) values. Skipping rows with any NaN.", input_file)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

    # Remove duplicates in timestamps.
    df.drop_duplicates(subset=INPUT_COL0_TS, keep='first',
                       inplace=True, ignore_index=True)

    # Freeze column is supposed to be binary (0 or 1)
    if df[INPUT_COL2_FREEZE].min() < 0 or df[INPUT_COL2_FREEZE].max() > 1:
        logger.warning(
            "Invalid values in input file(%s). Column 3 (freeze) value outside bounds (should be 0 or 1)",
            input_file
        )
        return False, None

    cast_to_type = {
        INPUT_COL0_TS: float,
        INPUT_COL1_MI: float,
        INPUT_COL2_FREEZE: int,
    }

    df = df.astype(cast_to_type)

    return True, df


def get_timeshift_from_input_file(input_file):
    global logger

    timeshift_val = None
    num_rows_processed = 0
    with open(input_file, "r") as read_obj:
        csv_reader = reader(read_obj)
        row1 = next(csv_reader)
        if (
            row1
            and len(row1) >= 2
            and (row1[0] == TIMESHIFT_HEADER or row1[0] == TIMESHIFT_HEADER_ALT)
        ):
            num_rows_processed += 1
            try:
                timeshift_val = float(row1[1])
            except ValueError:
                logger.error(
                    "Timeshift value (%s) is not numerical, ignoring it!", row1[1]
                )

    return timeshift_val, num_rows_processed


def get_input_dir():
    return input_dir


def set_input_dir(dir):
    global input_dir

    input_dir = dir


def select_input_dir(app):
    global input_dir

    open_folder = "."
    if input_dir:
        open_folder = os.path.dirname(input_dir)
    input_dir_temp = app.select_folder(folder=open_folder)
    if not input_dir_temp:
        logger.debug("no input folder selected, skipping")
        return

    input_dir = os.path.normpath(input_dir_temp)
    return input_dir


# Returns an output folder and create the dir, if needed.
# If an output dir is specified, use it.
# Else, output folder is '<parent of input file or folder>\output', create it
def get_output_dir(input_dir, output_dir, separate_files: bool):
    if output_dir:
        return output_dir
    output_folder = os.path.dirname(input_dir)
    base_name = os.path.basename(input_dir)
    if separate_files:
        output_folder = os.path.join(output_folder, base_name)
    else:
        output_folder = os.path.join(
            output_folder, base_name + OUTPUT_DIR_NAME)
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

    return output_folder


def open_file(filename):
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, filename])


def str_is_float(x: str) -> bool:
    try:
        float(x)
    except ValueError:
        return False
    return True


class CommonTetsMethods(object):
    def compare_csv_files(self, expected_csv_file, actual_file):
        logger.info(
            "\nComparing output file with expected.\n\tExpected: %s,\n\tOutput:%s",
            expected_csv_file,
            actual_file,
        )
        with open(expected_csv_file, "r") as t1, open(actual_file, "r") as t2:
            expected_lines = t1.readlines()
            output_lines = t2.readlines()
            x = 0
            for expected_line in expected_lines:
                expected_line_w = expected_line.strip().split(",")
                output_line_w = output_lines[x].strip().split(",")
                self.assertEqual(len(expected_line_w), len(output_line_w))
                for exp_w, actual_w in zip(expected_line_w, output_line_w):
                    if str_is_float(exp_w):
                        self.assertTrue(str_is_float(actual_w))
                        self.assertAlmostEqual(
                            float(exp_w),
                            float(actual_w),
                            2,
                            "output does not match",
                        )
                    else:
                        self.assertEqual(exp_w, actual_w)
                x += 1


class loghandler(logging.StreamHandler):
    """
    Custom logging handler
    """

    def __init__(self):
        self.result_log_ui_box = None
        logging.StreamHandler.__init__(self=self)

    def set_result_log_ui_box(self, result_log_ui_box):
        self.result_log_ui_box = result_log_ui_box

    def emit(self, record):
        """
        Writes the message to the output file (or the default logger stream),
        stdout and the UI result text box
        """
        try:
            msg = self.format(record)
            match record.levelno:
                case logging.WARNING:
                    p_msg = bcolors.WARN + msg + bcolors.ENDC
                case logging.CRITICAL | logging.ERROR:
                    p_msg = bcolors.ERR + msg + bcolors.ENDC
                case _:
                    p_msg = msg
            if self.result_log_ui_box is not None:
                match record.levelno:
                    case logging.WARNING:
                        # If it wasn't previously set to higher attention.
                        if not self.result_log_ui_box.bg == LIGHT_RED_COLOR:
                            self.result_log_ui_box.bg = LIGHT_YELLOW_COLOR
                    case logging.CRITICAL | logging.ERROR:
                        self.result_log_ui_box.bg = LIGHT_RED_COLOR
                self.result_log_ui_box.value += msg
            print(p_msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)
