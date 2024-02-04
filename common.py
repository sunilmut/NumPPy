#!/usr/bin/python

import os
import subprocess
import sys
from csv import reader

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

# globals
logger = None
input_dir = ""


def parse_input_file_into_df(input_file, skip_num_initial_rows):
    """
    Parse the input file
    returns bool, dataframe
    bool - True if parsing was successful; False otherwise
    dataframe - Parsed dataframe
    """
    in_col_names = [INPUT_COL0_TS, INPUT_COL1_MI, INPUT_COL2_FREEZE]
    df = pd.read_csv(input_file, names=in_col_names,
                     skiprows=skip_num_initial_rows)

    # Do some basic format checking. All input fields are expected
    # to be numeric in nature.
    if not (
        is_numeric_dtype(df[INPUT_COL0_TS])
        and is_numeric_dtype(df[INPUT_COL1_MI])
        and is_numeric_dtype(df[INPUT_COL2_FREEZE])
    ):
        print("Invalid input file format: " + input_file)
        return False, pd.DataFrame()

    # Freeze column is supposed to be binary (0 or 1)
    if df[INPUT_COL2_FREEZE].min() < 0 or df[INPUT_COL2_FREEZE].max() > 1:
        print(
            "Invalid input file format in "
            + input_file
            + ". Column 3 (freeze) value outside bounds (should be 0 or 1)"
        )
        return False, pd.DataFrame()

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
