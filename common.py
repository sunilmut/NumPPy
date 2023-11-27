#!/usr/bin/python

import sys
import pandas as pd
from pandas.api.types import is_numeric_dtype
from csv import reader
import os
import subprocess
import numpy as np
import glob

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
def get_output_dir(input_dir, output_dir):
    if output_dir:
        return output_dir
    output_folder = os.path.dirname(input_dir)
    base_name = os.path.basename(input_dir)
    output_folder = os.path.join(output_folder, base_name + OUTPUT_DIR_NAME)
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    return output_folder

def open_file(filename):
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, filename])

def get_param_file_from_name(param_name):
    param_file = os.path.join(get_input_dir(), Parameters.PARAMETERS_DIR_NAME, param_name)
    return param_file + CSV_EXT

class Parameters:
    PARAM_TIME_WINDOW_START_LIST = "Start_Timestamp_List"
    PARAM_TIME_WINDOW_DURATION = "Window_Duration_In_Sec"
    PARAMETERS_DIR_NAME = "parameters"
    _param_col_names = [PARAM_TIME_WINDOW_START_LIST, PARAM_TIME_WINDOW_DURATION]

    def __init__(self):
        # currently selected parameter
        self._cur_selected_param = ""
        self._param_name_list = []
        # Parameter values as dataframe. There is one dataframe for each parameter
        self._param_df_list = []
        self._param_window_duration = 0
        self._param_start_timestamp_series = pd.Series(dtype=np.float64)
        self._param_dir = ""

    def reset(self):
        self._cur_selected_param = ""
        self._param_name_list = []
        self._param_df_list = []
        self._param_window_duration = 0
        self._param_start_timestamp_series = pd.Series(dtype=np.float64)
        self._param_dir = ""

    def parse(self, input_dir):
        self.reset()
        self._param_dir = os.path.join(input_dir, Parameters.PARAMETERS_DIR_NAME)
        param_dir = self._param_dir
        logger.debug("Parameters dir is %s", self._param_dir)
        if not os.path.isdir(param_dir):
            raise ValueError("Parameter folder " + param_dir + " does not exist!")

        search_path = os.path.join(param_dir, "*.csv")
        for param_file in glob.glob(search_path):
            logger.debug("param file: %s", param_file)
            if not os.path.isfile(param_file):
                continue

            param_file_name_without_ext = os.path.splitext(os.path.basename(param_file))[0]
            self._param_name_list.append(param_file_name_without_ext)
            param_df = pd.read_csv(param_file, names=Parameters.get_param_column_names(), header=None, skiprows=1)
            self._param_df_list.append(param_df)

        if len(self._param_name_list) > 0:
            self._cur_selected_param = self._param_name_list[0]

        return

    def get_currently_selected_param(self):
        return self._cur_selected_param

    def set_currently_selected_param(self, param_name):
        if param_name in self._param_name_list:
            self._cur_selected_param = param_name
        else:
            raise ValueError("Parameter name " + param_name + " is not part of valid parameter list!")

    def get_param_name_list(self):
        return self._param_name_list

    def get_param_values(self, param_name):
        try:
            param_index = self._param_name_list.index(param_name)
        except ValueError:
            logger.error("Parameter value: %s is out of index", param_name)
            return

        param_df = self._param_df_list[param_index]
        return Parameters.parse_param_df(param_df)

    def get_param_file_from_name(self, param_name):
        return os.path.join(self._param_dir, param_name + CSV_EXT)

    @staticmethod
    def parse_param_df(df):
        value = df[Parameters.PARAM_TIME_WINDOW_DURATION].iat[0]
        w_duration = 0
        if not pd.isnull(value):
            w_duration = value

        ts_series = df[Parameters.PARAM_TIME_WINDOW_START_LIST]
        ts_series.sort_values(ascending=True)

        return w_duration, ts_series

    @staticmethod
    def get_param_column_names():
        return Parameters._param_col_names