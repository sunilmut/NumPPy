#!/usr/bin/python

import sys
import getopt
import logging
import os
import pandas as pd
from pandas.api.types import is_numeric_dtype
import glob
import guizero
from guizero import App, Box, CheckBox, Combo, ListBox, PushButton, Text, TextBox, TitleBox, Window
import subprocess
import numpy as np
import csv
from csv import reader

# Constants:
# Number of initial rows to skip.
NUM_INITIAL_ROWS_TO_SKIP = 3

ZERO_TO_ONE = "0 to 1"
ONE_TO_ZERO = "1 to 0"

# input coloumns
INPUT_COL0 = "timestamps"
INPUT_COL1 = "Motion Index"
INPUT_COL2 = "Freeze"

# output columns
OUTPUT_COL0_TS = "timestamps"
OUTPUT_COL1_MI = "Motion Index"
OUTPUT_COL2_MI_AVG = "Avg of MI"
OUTPUT_COL3_FREEZE_TP = "Freezing TurnPoints"

# output directory and file names
OUTPUT_DIR_NAME = "_output"
PARAMETERS_DIR_NAME = "parameters"
OUTPUT_BASE = "_output_base.csv"
CSV = ".csv"
UNDERSCORE = "_"
OUTPUT_ZERO_TO_ONE_CSV_NAME = "_01_OnsetT"
OUTPUT_ZERO_TO_ONE_CSV_NAME_TS = "_01_OnsetT_ts"
OUTPUT_ZERO_TO_ONE_NOP_CSV_NAME = "_01_NOp"
OUTPUT_ZERO_TO_ONE_NOP_CSV_NAME_TS = "_01_NOp_ts"
OUTPUT_ONE_TO_ZERO_CSV_NAME = "_10_OnsetT"
OUTPUT_ONE_TO_ZERO_CSV_NAME_TS = "_10_OnsetT_ts"
OUTPUT_ONE_TO_ZERO_NOP_CSV_NAME = "_10_NOp"
OUTPUT_ONE_TO_ZERO_NOP_CSV_NAME_TS = "_10_NOp_ts"

# UI related constants
INPUT_FOLDER_NAME_BOX_MAX_WIDTH = 26
PARAM_TS_CRITERIA = "Criteria_Timestamp_In_Sec"
PARAM_TIME_WINDOW_START_LIST = "Start_Timestamp_List"
PARAM_TIME_WINDOW_DURATION = "Window_Duration_In_Sec"
PARAM_UI_TIME_WINDOW_START_TIMES = "Time window start (secs):"
PARAM_UI_MIN_TIME_DURATION_CRITERIA_TEXT = "Min time duration criteria (secs): "
PARAM_UI_TIME_WINDOW_DURATION_TEXT = "Time window duration (secs): "

# globals
output_dir = ""
out_col_names = [OUTPUT_COL0_TS, OUTPUT_COL1_MI,
                 OUTPUT_COL2_MI_AVG, OUTPUT_COL3_FREEZE_TP]
out_file_zero_to_one = ""
out_file_zero_to_one_ts = ""
out_file_zero_to_one_un = ""
out_file_zero_to_one_un_ts = ""
out_file_one_to_zero = ""
out_file_one_to_zero_ts = ""
out_file_one_to_zero_un = ""
out_file_one_to_zero_un_ts = ""
param_col_names = [PARAM_TS_CRITERIA,
                   PARAM_TIME_WINDOW_START_LIST, PARAM_TIME_WINDOW_DURATION]
param_min_time_duration = 0
param_window_duration = 0
param_start_timestamp_series = pd.Series(dtype=np.float64)
input_dir = ""

# Arrays to store the parameter names and its value as a dataframe
# There is a dataframe value for each parameter and the indexes
# for these two arrays should be kept in sync.
# Parameter names
param_name_list = []
# Parameter values as dataframe. There is one dataframe for each parameter
param_df_list = []
# Currently selected parameter name
cur_selected_param = None

# When set to True, only the currently selected parameter is processed instead
# of all the parameters. Default is to process all parameters.
only_process_cur_param = False

# Timeshift header in the input
TIMESHIFT_HEADER = "timeshift"


def apply_duration_criteria(ts_series, param_min_time_duration):
    it = ts_series.iteritems()
    prev_ts = 0
    for idx, val in it:
        if (val - prev_ts >= param_min_time_duration):
            yield idx
        prev_ts = val


def apply_timewindow_filter(ts_series, timstamp_filter_series, duration):
    it = ts_series.iteritems()
    for idx, val in it:
        for filter_idx, filter_val in timstamp_filter_series.items():
            if val < filter_val:
                break
            if val >= filter_val and val <= filter_val + duration:
                yield idx
                break


def split_df_and_output(out_df, timeshift_val, out_file_zero_to_one, out_file_one_to_zero,
                        out_file_zero_to_one_ts, out_file_one_to_zero_ts):
    """
    Split the dataframe based on whether it is a [0->1] or [1->0] transitions
    and output to respective files.
    Timestamps file should have the header
    """
    if out_df.empty:
        return

    output_df = out_df[:]
    output_df[OUTPUT_COL0_TS] = output_df[OUTPUT_COL0_TS] + timeshift_val
    out_zero_to_one_df = pd.DataFrame(columns=out_col_names)
    out_one_to_zero_df = pd.DataFrame(columns=out_col_names)
    out_zero_to_one_df = output_df.loc[output_df[OUTPUT_COL3_FREEZE_TP]
                                       == ZERO_TO_ONE]
    out_zero_to_one_ts_df = out_zero_to_one_df.loc[:, OUTPUT_COL0_TS]
    out_one_to_zero_df = output_df.loc[output_df[OUTPUT_COL3_FREEZE_TP]
                                       == ONE_TO_ZERO]
    out_one_to_zero_ts_df = out_one_to_zero_df.loc[:, OUTPUT_COL0_TS]
    out_zero_to_one_df.to_csv(out_file_zero_to_one, index=False)
    out_zero_to_one_ts_df.to_csv(
        out_file_zero_to_one_ts, index=False, header=True)
    out_one_to_zero_df.to_csv(out_file_one_to_zero, index=False)
    out_one_to_zero_ts_df.to_csv(
        out_file_one_to_zero_ts, index=False, header=True)


def out_base(input_file, output_folder):
    input_file_without_ext = os.path.splitext(os.path.basename(input_file))[0]
    return os.path.join(
        output_folder, input_file_without_ext + OUTPUT_BASE
    )


def format_out_nop_file_name(input_file, param_names, output_folder):
    """
    Format the 'NOp' output file name
    """
    global out_file_zero_to_one_un, out_file_zero_to_one_un_ts
    global out_file_one_to_zero_un, out_file_one_to_zero_un_ts

    input_file_without_ext = os.path.splitext(os.path.basename(input_file))[0]
    out_file_zero_to_one_un = os.path.join(
        output_folder, input_file_without_ext +
        OUTPUT_ZERO_TO_ONE_NOP_CSV_NAME + param_names + CSV
    )
    out_file_zero_to_one_un_ts = os.path.join(
        output_folder, input_file_without_ext +
        OUTPUT_ZERO_TO_ONE_NOP_CSV_NAME_TS + param_names + CSV
    )
    out_file_one_to_zero_un = os.path.join(
        output_folder, input_file_without_ext +
        OUTPUT_ONE_TO_ZERO_NOP_CSV_NAME + param_names + CSV
    )
    out_file_one_to_zero_un_ts = os.path.join(
        output_folder, input_file_without_ext +
        OUTPUT_ONE_TO_ZERO_NOP_CSV_NAME_TS + param_names + CSV
    )

    print("\nInput file: ", os.path.basename(os.path.basename(input_file)))
    print("Output file:")
    print("\tNOp [0->1]: ", os.path.basename(out_file_zero_to_one_un))
    print("\tNOp [1->0]: ", os.path.basename(out_file_one_to_zero_un))
    print("\tNOp [0->1] TimeStamps Only:: ",
          os.path.basename(out_file_zero_to_one_un_ts))
    print("\tNOp [1->0] TimeStamps Only:: ",
          os.path.basename(out_file_one_to_zero_un_ts))


def format_out_file_names(input_file, param_name, output_folder):
    """
    Format the output file names
    """
    global out_file_zero_to_one, out_file_zero_to_one_ts
    global out_file_one_to_zero, out_file_one_to_zero_ts

    param_ext = ""
    if param_name:
        param_ext = "_" + param_name

    input_file_without_ext = os.path.splitext(os.path.basename(input_file))[0]
    out_file_zero_to_one = os.path.join(
        output_folder, input_file_without_ext +
        OUTPUT_ZERO_TO_ONE_CSV_NAME + param_ext + CSV
    )
    out_file_zero_to_one_ts = os.path.join(
        output_folder, input_file_without_ext +
        OUTPUT_ZERO_TO_ONE_CSV_NAME_TS + param_ext + CSV
    )
    out_file_one_to_zero = os.path.join(
        output_folder, input_file_without_ext +
        OUTPUT_ONE_TO_ZERO_CSV_NAME + param_ext + CSV
    )
    out_file_one_to_zero_ts = os.path.join(
        output_folder, input_file_without_ext +
        OUTPUT_ONE_TO_ZERO_CSV_NAME_TS + param_ext + CSV
    )

    print("\nInput file: ", os.path.basename(os.path.basename(input_file)))
    print("Output file:")
    print("\t[0->1]: ", os.path.basename(out_file_zero_to_one))
    print("\t[1->0]: ", os.path.basename(out_file_one_to_zero))
    print("\t[0->1] TimeStamps Only: ",
          os.path.basename(out_file_zero_to_one_ts))
    print("\t[1->0] TimeStamps Only: ",
          os.path.basename(out_file_one_to_zero_ts))


def parse_input_file_into_df(input_file, skip_num_initial_rows):
    """
    Parse the input file
    returns bool, dataframe
    bool - True if parsing was successful; False otherwise
    dataframe - Parsed dataframe
    """
    in_col_names = [INPUT_COL0, INPUT_COL1, INPUT_COL2]
    df = pd.read_csv(input_file, names=in_col_names,
                     skiprows=skip_num_initial_rows)

    # Do some basic format checking. All input fields are expected
    # to be numeric in nature.
    if not (
        is_numeric_dtype(df[INPUT_COL0])
        and is_numeric_dtype(df[INPUT_COL1])
        and is_numeric_dtype(df[INPUT_COL2])
    ):
        print("Invalid input file format: " + input_file)
        return False, pd.DataFrame()

    # Freeze column is supposed to be binary (0 or 1)
    if df[INPUT_COL2].min() < 0 or df[INPUT_COL2].max() > 1:
        print(
            "Invalid input file format in "
            + input_file
            + ". Column 3 (freeze) value outside bounds (should be 0 or 1)"
        )
        return False, pd.DataFrame()

    return True, df


def parse_param_folder():
    """
    Parse parameter folder and create a list of parameter dataframe(s)
    out of it.
    """
    global input_dir, param_name_list, param_df_list

    param_folder = os.path.join(input_dir, "parameters")
    if not os.path.isdir(param_folder):
        return False, ("Parameter folder " + param_folder + " does not exist!")

    search_path = param_folder + "\*.csv"

    param_name_list.clear()
    param_df_list.clear()
    for param_file in glob.glob(search_path):
        if not os.path.isfile(param_file):
            continue

        param_file_name_without_ext = os.path.splitext(
            os.path.basename(param_file))[0]
        param_name_list.append(param_file_name_without_ext)
        param_df = pd.read_csv(
            param_file, names=param_col_names, header=None, skiprows=1)
        param_df_list.append(param_df)

    print(param_name_list)
    print(param_df_list)

    return True, ""


def parse_all_param_dfs():
    global param_df_list

    for df in param_df_list:
        t_duration, w_duration, ts_series = parse_param_df(df)


def parse_param_df(df):
    value = df[PARAM_TS_CRITERIA].iat[0]
    if not pd.isnull(value):
        t_duration = value
        print("Parameter - time duration: ", t_duration)

    value = df[PARAM_TIME_WINDOW_DURATION].iat[0]
    if not pd.isnull(value):
        w_duration = value
        print("Parameter - window duration: ", w_duration)

    ts_series = df[PARAM_TIME_WINDOW_START_LIST]
    ts_series.sort_values(ascending=True)

    return t_duration, w_duration, ts_series


def get_param_file_from_name(param_name):
    global input_dir

    param_file = os.path.join(input_dir, PARAMETERS_DIR_NAME)
    param_file = os.path.join(param_file, param_name)

    return param_file + CSV


def parse_cur_param_file():
    parse_param(cur_selected_param)


def parse_param(cur_selected_param):
    """
    Parse the paramter file
    """
    global param_min_time_duration
    global param_window_duration, param_start_timestamp_series
    global input_dir

    if not cur_selected_param:
        return

    param_min_time_duration = 0
    param_window_duration = 0
    param_start_timestamp_series = pd.Series(dtype=np.float64)

    try:
        param_index = param_name_list.index(cur_selected_param)
    except ValueError:
        logging.error("Parameter value: %s is out of index",
                      cur_selected_param)
        return

    param_df = param_df_list[param_index]
    param_min_time_duration, param_window_duration, param_start_timestamp_series = parse_param_df(
        param_df)

    refresh_param_values(param_start_timestamp_series)


def print_df(msg, df):
    """
    Print a dataframe with the message
    """
    if df.empty:
        return

    logging.debug(msg)
    logging.debug(df)


def process_input_file(input_file, output_folder):
    """
    Main logic routine to parse the input and spit out the output
    """
    global out_file_zero_to_one, out_file_zero_to_one_un
    global out_file_zero_to_one_ts, out_file_zero_to_one_un_ts
    global out_file_one_to_zero, out_file_one_to_zero_un
    global out_file_one_to_zero_ts, out_file_one_to_zero_un_ts
    global param_min_time_duration, param_window_duration, param_start_timestamp_series

    timeshift_val, num_rows_processed = get_timeshift_from_input_file(
        input_file)

    if timeshift_val:
        print("Applying a timeshift value of " +
              str(timeshift_val) + " on input file " + input_file)
    else:
        timeshift_val = 0

    # Parse the input file
    success, df = parse_input_file_into_df(
        input_file, NUM_INITIAL_ROWS_TO_SKIP + num_rows_processed)
    if not success:
        return

    # Parse all the parameter files.
    parse_param_folder()

    sum = 0
    itr = 0
    prev_idx = 0
    # p.s - start with a freeze value of 1 as that will make the very first row
    #        with freeze value of 0 look like a transition to avoid special
    #        case handle for row 0.
    prev_freeze = 1
    out_df = pd.DataFrame(columns=out_col_names)
    freeze_0_ts = 0
    freeze_0_mi = 0

    # Iterate over all the rows
    for (idx, row) in df.iterrows():
        # All row's with freeze value of '0' are valuable and need to
        # be summed up (until a transition)
        if row.values[2] == 0:
            sum += row.values[1]
            itr += 1

        # On transition from [1->0], record the row values which will
        # be later used for output
        if row.values[2] != prev_freeze and row.values[2] == 0:
            freeze_0_ts = row.values[0]
            freeze_0_mi = row.values[1]

        # For output, we only care about rows where there is a transition
        # of freeze value from [0->1]
        if row.values[2] != prev_freeze and row.values[2] == 1:
            # Divide by zero exceptional condition. Capture the details
            # for troubleshooting.
            if itr == 0:
                logging.error("Current index: %s", str(idx + 4))
                logging.error("Row value: %s", row.to_string())
                logging.error("Previous index: %s", str(prev_idx + 4))
                return False

            # On transition from [0->1], we need to capture two entries:
            # One for the row which has the freeze value of 0 (that was previously captured)
            # and for the row with the freeze value of 1 (current idx)
            df_o = pd.DataFrame(
                {
                    OUTPUT_COL0_TS: freeze_0_ts,
                    OUTPUT_COL1_MI: freeze_0_mi,
                    OUTPUT_COL2_MI_AVG: [sum / itr],
                    OUTPUT_COL3_FREEZE_TP: [ONE_TO_ZERO],
                }
            )
            out_df = pd.concat([out_df, df_o], ignore_index=True, sort=False)
            df_o = pd.DataFrame(
                {
                    OUTPUT_COL0_TS: [row.values[0]],
                    OUTPUT_COL1_MI: [row.values[1]],
                    OUTPUT_COL2_MI_AVG: [sum / itr],
                    OUTPUT_COL3_FREEZE_TP: [ZERO_TO_ONE],
                }
            )
            out_df = pd.concat([out_df, df_o], ignore_index=True, sort=False)

            # On transition, reset the values.
            sum = 0
            itr = 0
            prev_idx = idx
            freeze_0_ts = 0
            freeze_0_mi = 0

        prev_freeze = row.values[2]

    if out_df.empty:
        return

    print_df("After initial parsing (without any criteria or filter)", out_df)
    out_df.to_csv(out_base(input_file, output_folder), index=False)

    # 'NOp' -> no parameter. This is the left over of the original data
    # after all the parameters have been processed. 'NOp' starts with
    # the original set and as each parameter gets process the set
    # gets subtracted by that.
    # Make a copy of out dataframe 'by value'
    nop_df = out_df[:]
    params_name = ""

    # Iterate through the parameters and apply each one of them
    for idx, param_name in enumerate(param_name_list):
        if only_process_cur_param and param_name != cur_selected_param:
            continue

        # Make a copy by value
        temp_out_df = out_df[:]
        params_name += UNDERSCORE + param_name
        param_min_time_duration, param_window_duration, param_start_timestamp_series = parse_param_df(
            param_df_list[idx])

        # Format the file names of the output files using the parameter name
        format_out_file_names(input_file, param_name, output_folder)

        # Apply any minimum time duration criteria
        if param_min_time_duration > 0:
            temp_out_df = temp_out_df.loc[list(apply_duration_criteria(
                temp_out_df.iloc[:, 0], param_min_time_duration))]
            print_df("After applying min time duration criteria", temp_out_df)

        # Apply time window filter
        if not param_start_timestamp_series.empty:
            filter_list = list(apply_timewindow_filter(
                temp_out_df.iloc[:, 0], param_start_timestamp_series, param_window_duration))
            temp_out_df = temp_out_df.loc[filter_list]
            print_df("After applying timestamp filter", temp_out_df)

        nop_df = pd.merge(temp_out_df, nop_df, how='outer', indicator=True).query(
            "_merge != 'both'").drop('_merge', axis=1).reset_index(drop=True)

        split_df_and_output(temp_out_df, timeshift_val, out_file_zero_to_one, out_file_one_to_zero,
                            out_file_zero_to_one_ts, out_file_one_to_zero_ts)

        print("After processing param" + param_name)
        print(nop_df)

    if not nop_df.empty:
        format_out_nop_file_name(input_file, params_name, output_folder)
        split_df_and_output(nop_df, timeshift_val,
                            out_file_zero_to_one_un, out_file_one_to_zero_un,
                            out_file_zero_to_one_un_ts, out_file_one_to_zero_un_ts)

    return True


def print_help():
    """
    Display help
    """
    print("\nHelp/Usage:\n")
    print(
        "python motion.py -i <input folder or .csv file> -d <output_directory> -v -h\n"
    )
    print("where:")
    print("-i (required): input folder or .csv file.")
    print("-d (optional): output folder.")
    print("-v (optional): run in verbose mode")
    print("-h (optional): print this help")
    print("\nExamples:\n")
    print("\nProcess input file:")
    print("\tpython motion.py -i input.csv")
    print("\nProcess all the csv files from the input folder:")
    print("\tpython motion.py -i c:\\data\\input")
    print(
        "\nProcess all the csv files from the input folder and use the output folder:"
    )
    print("\tpython motion.py -i c:\\data\\input -d c:\\data\\output")
    print("\nNotes:")
    print("\tClose the output file prior to running.")
    sys.exit()


def get_inpput_files(input_dir):
    input_files = []

    # Normalize the path to deal with backslash/frontslash
    input_folder_or_file = os.path.normpath(input_dir)
    search_path = os.path.join(input_folder_or_file, "*.csv")
    for file in glob.glob(search_path):
        input_files.append(file)

    return input_files


def main(argv, input_folder_or_file):
    global input_dir

    input_files = []
    output_folder = ""

    try:
        opts, args = getopt.getopt(argv, "vhi:o:d:h:")
    except getopt.GetoptError:
        print_help()
    for opt, arg in opts:
        if opt == "-h":
            print_help()
        elif opt in ("-i"):
            input_folder_or_file = arg
        elif opt in ("-d"):
            output_folder = arg
        elif opt in ("-v"):
            logging.basicConfig(level=logging.DEBUG)
        else:
            print_help()

    # strip the quotes at the start and end, else
    # paths with white spaces won't work.
    input_folder_or_file = input_folder_or_file.strip('"')
    print(input_folder_or_file)

    # get the input file if one is not provided.
    if not input_folder_or_file:
        input_folder_or_file = input(
            "Provide an input folder or .csv file name: ")

    if os.path.isdir(input_folder_or_file):
        input_dir = input_folder_or_file
        input_files = get_inpput_files(input_dir)
    elif os.path.isfile(input_folder_or_file):
        input_dir = os.path.dirname(input_folder_or_file)
        input_files.append(input_folder_or_file)
    else:
        print("The input path is not a valid directory or file: ",
              input_folder_or_file)
        print_help()

    # If an output folder is specified, use it.
    # Else, output folder is '<parent of input file or folder>\output', create it
    if not output_folder:
        output_folder = os.path.dirname(input_folder_or_file)
        base_name = os.path.basename(input_folder_or_file)
        output_folder = os.path.join(
            output_folder, base_name + OUTPUT_DIR_NAME)
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

    print("\nInput folder: ", input_dir)
    print("Output folder: ", output_folder)
    successfully_parsed_files = []
    unsuccessfully_parsed_files = []
    for input_file in input_files:
        parsed = process_input_file(input_file, output_folder)
        if parsed:
            successfully_parsed_files.append(input_file)
        else:
            unsuccessfully_parsed_files.append(input_file)

    return output_folder, successfully_parsed_files, unsuccessfully_parsed_files


def get_timeshift_from_input_file(input_file):
    timeshift_val = None
    num_rows_processed = 0
    with open(input_file, 'r') as read_obj:
        csv_reader = reader(read_obj)
        row1 = next(csv_reader)
        if row1 and len(row1) > 2 and row1[0] == TIMESHIFT_HEADER:
            num_rows_processed += 1
            try:
                timeshift_val = int(row1[1])
                print("timeshift val is " + str(timeshift_val))
            except ValueError:
                print("timeshift value is not an integer in file" + input_file)

    return timeshift_val, num_rows_processed


"""
------------------------------------------------------------
                UI related stuff
------------------------------------------------------------
"""


def parse_input_file_for_timeshift(input_dir):
    input_files = get_inpput_files(input_dir)
    for input_file in input_files:
        timeshift_val = get_timeshift_from_input_file(
            input_file)[0]
        if not timeshift_val:
            app.warn(
                "Uh oh!", "Input file " + input_file + " does not has a timeshift value or a timeshift value that is not an integer")


def line():
    Text(app, "------------------------------------------------------------------------------------------------------")


def select_input_folder():
    global input_dir, param_name_list, cur_selected_param

    input_dir_temp = app.select_folder()
    if not input_dir_temp:
        return

    input_dir = input_dir_temp
    parse_input_file_for_timeshift(input_dir)
    input_folder_text_box.value = os.path.basename(input_dir)
    input_folder_text_box.width = min(
        len(input_folder_text_box.value), INPUT_FOLDER_NAME_BOX_MAX_WIDTH)
    parse_param_folder()
    refresh_param_names_combo_box(param_name_list)
    if len(param_name_list):
        print("parsing parameter: " + param_name_list[0])
        cur_selected_param = param_name_list[0]
        parse_param(cur_selected_param)


def open_output_folder():
    global output_dir
    if not output_dir:
        app.warn(
            "Uh oh!", "Output folder dose not exist! Select input folder and process first.")
        return

    # Normalize the path to deal with backslash/frontslash
    subprocess.Popen(f'explorer /open,{os.path.normpath(output_dir)}')


def open_params_file():
    if not cur_selected_param:
        return

    param_file = get_param_file_from_name(cur_selected_param)
    if not os.path.isfile(param_file):
        app.warn(
            "Uh oh!", "Parameters file " + param_file + " is not a file!")
        return

    os.startfile(param_file)


def refresh_result_text_box(successfully_parsed_files, unsuccessfully_parsed_files):
    result_success_list_box.clear()
    for val in successfully_parsed_files:
        result_success_list_box.append(val)


def process():
    global input_dir, output_dir

    if not input_dir:
        app.warn(
            "Uh oh!", "No input folder specified. Please select an input folder and run again!")
        return

    output_dir, successfully_parsed_files, unsuccessfully_parsed_files = main(
        sys.argv[1:], input_dir)
    refresh_result_text_box(successfully_parsed_files,
                            unsuccessfully_parsed_files)
    # result_window.show()


def select_param(selected_param_value):
    """
    This method is called when the user selects a parameter from the parameter
    name drop down box.
    """
    global cur_selected_param

    if not selected_param_value:
        return

    cur_selected_param = selected_param_value
    try:
        param_index = param_name_list.index(cur_selected_param)
    except ValueError:
        logging.error("Parameter value: %s is out of index",
                      cur_selected_param)
        return

    df = param_df_list[param_index]
    param_min_time_duration, param_window_duration, param_start_timestamp_series = parse_param_df(
        df)

    refresh_param_values(param_start_timestamp_series)


def refresh_param_names_combo_box(param_name_list):
    param_names_combo_box.clear()
    for param_name in param_name_list:
        param_names_combo_box.append(param_name)

    param_names_combo_box.show()


def refresh_param_values(param_start_timestamp_series):
    set_min_time_duration_box_value()
    set_time_window_duration_box_value()
    refresh_ts_list_box(param_start_timestamp_series)


def refresh_ts_list_box(ts_series):
    ts_series_list_box.clear()
    for val in ts_series:
        ts_series_list_box.append(val)


def set_min_time_duration_box_value():
    global param_min_time_duration
    min_time_duration_box.value = param_min_time_duration


def set_time_window_duration_box_value():
    global param_window_duration
    time_window_duration_box.value = param_window_duration


def only_process_sel_param():
    global only_process_cur_param

    if only_process_cur_param_box.value == 1:
        only_process_cur_param = True
    else:
        only_process_cur_param = False


if __name__ == "__main__":
    """
    Main entry point
    """
    # Main app
    app = App("",  height=750, width=800)

    # Title box
    titlebox = TitleBox(app, "")
    titlebox.bg = "white"
    title = Text(app, text="Motion Data Processing App",
                 size=16, font="Arial Bold", width=25)
    title.bg = "white"
    line()

    # Select input folder button
    Text(app, "Select Input Folder --> Check Parameters --> Process",
         font="Verdana bold")
    line()
    input_folder_button = PushButton(
        app, command=select_input_folder, text="Input Folder", width=26)
    input_folder_button.tk.config(font=("Verdana bold", 14))

    line()
    input_folder_text_box = TextBox(app)
    # Non editable
    input_folder_text_box.disable()
    input_folder_text_box.width = INPUT_FOLDER_NAME_BOX_MAX_WIDTH
    input_folder_text_box.font = "Verdana bold"
    input_folder_text_box.text_size = 14
    line()

    # Master parameter box
    param_box = Box(app, layout="grid")
    cnt = 0
    param_title_box = TitleBox(param_box, text="", grid=[0, cnt])
    param_title = Text(param_title_box, text="Parameters",
                       size=10, font="Arial Bold", align="left", color="orange")

    # Parameter names combo box (grid high to keep it on the right side)
    param_names_combo_box = Combo(
        param_box, options=[], grid=[10, cnt], command=select_param)
    param_names_combo_box.clear()
    param_names_combo_box.text_color = "orange"
    # param_names_combo_box.hide()
    cnt += 1

    # Boxes related to showing parameters
    min_time_duration_label_box = Text(
        param_box, text=PARAM_UI_MIN_TIME_DURATION_CRITERIA_TEXT, grid=[0, cnt], align="left")
    min_time_duration_box = TextBox(
        param_box, text="", grid=[1, cnt], align="left")
    # Non editable
    min_time_duration_box.disable()
    cnt += 1

    time_window_duration_label_box = Text(
        param_box, text=PARAM_UI_TIME_WINDOW_DURATION_TEXT, grid=[0, cnt], align="left")
    time_window_duration_box = TextBox(
        param_box, text="", grid=[1, cnt], align="left")
    # Non editable
    time_window_duration_box.disable()
    cnt += 1

    ts_series_list_label_box = Text(param_box, text=PARAM_UI_TIME_WINDOW_START_TIMES,
                                    grid=[0, cnt], align="left")
    ts_series_list_box = ListBox(
        param_box, param_start_timestamp_series, scrollbar=True, grid=[1, cnt], align="left")
    cnt += 1

    # Open & update parameters file button
    center_box = Box(app, layout="grid")
    open_params_button = PushButton(center_box, command=open_params_file,
                                    text="Open parameters file", grid=[0, 1], width=17, align="left")
    open_params_button.tk.config(font=("Verdana bold", 10))
    update_params_button = PushButton(center_box, text="Refresh",
                                      command=parse_cur_param_file, grid=[1, 1], width=17, align="left")
    update_params_button.tk.config(font=("Verdana bold", 10))
    line()

    # Process input button
    process_button = PushButton(app, text="Process", command=process, width=26)
    process_button.tk.config(font=("Verdana bold", 14))
    only_process_cur_param_box = CheckBox(
        app, text="Only process the selected parameter", command=only_process_sel_param)
    line()

    # Browse output folder button
    browse_output_folder_button = PushButton(
        app, command=open_output_folder, text="Browse output folder", width=20)
    browse_output_folder_button.tk.config(font=("Verdana bold", 10))
    line()

    # New Result window
    result_window = Window(app, title="Result Window",
                           height=500, width=500, visible=False)
    result_text_box = Text(result_window, text="")
    result_success_list_box = ListBox(result_window, [], scrollbar=True)
    result_window.hide()

    # Display the app
    app.display()
