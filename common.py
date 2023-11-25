#!/usr/bin/python

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_integer_dtype
from csv import reader

# input coloumns
INPUT_COL0_TS = "timestamps"
INPUT_COL1_MI = "Motion Index"
INPUT_COL2_FREEZE = "Freeze"

# Timeshift header in the input
TIMESHIFT_HEADER = "timeshift"
TIMESHIFT_HEADER_ALT = "shift"

# Number of initial rows to skip.
NUM_INITIAL_ROWS_TO_SKIP = 3

# globals
logger = None

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

    return True, df



def get_timeshift_from_input_file(input_file):
    global logger

    timeshift_val = None
    num_rows_processed = 0
    with open(input_file, 'r') as read_obj:
        csv_reader = reader(read_obj)
        row1 = next(csv_reader)
        if row1 and len(row1) >= 2 and (row1[0] == TIMESHIFT_HEADER or row1[0] == TIMESHIFT_HEADER_ALT):
            num_rows_processed += 1
            try:
                timeshift_val = float(row1[1])
            except ValueError:
                logger.error(
                    "Timeshift value (%s) is not numerical, ignoring it!", row1[1])

    return timeshift_val, num_rows_processed
