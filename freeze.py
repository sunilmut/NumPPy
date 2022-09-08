#!/usr/bin/python

import sys
import getopt
import logging
import os
import pandas as pd
from pandas.api.types import is_numeric_dtype
import glob
import guizero
from guizero import App, Box, ListBox, PushButton, Text, TitleBox, Window
import subprocess
import numpy as np

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
PARAMETERS_DIR_NAME = "parameters\param.csv"
OUTPUT_BASE = "_output_base.csv"
OUTPUT_ZERO_TO_ONE_CSV_NAME = "_01_OnsetT.csv"
OUTPUT_ZERO_TO_ONE_CSV_NAME_TS = "_01_OnsetT_ts.csv"
OUTPUT_ZERO_TO_ONE_UN_CSV_NAME = "_01_unspecified.csv"
OUTPUT_ZERO_TO_ONE_UN_CSV_NAME_TS = "_01_unspecified_ts.csv"
OUTPUT_ONE_TO_ZERO_CSV_NAME = "_10_OnsetT.csv"
OUTPUT_ONE_TO_ZERO_CSV_NAME_TS = "_10_OnsetT_ts.csv"
OUTPUT_ONE_TO_ZERO_UN_CSV_NAME = "_10_unspecified.csv"
OUTPUT_ONE_TO_ZERO_UN_CSV_NAME_TS = "_10_unspecified_ts.csv"

# Parameters column
PARAM_TS_CRITERIA = "Criteria_Timestamp_In_Sec"
PARAM_TIME_WINDOW_START_LIST = "Start_Timestamp_List"
PARAM_TIME_WINDOW_DURATION = "Window_Duration_In_Sec"
PARAM_UI_TIME_WINDOW_START_TIMES = "Time window start (secs):"
PARAM_UI_MIN_TIME_DURATION_CRITERIA_TEXT = "Min time duration criteria (secs): "
PARAM_UI_TIME_WINDOW_DURATION = "Time window duration (secs): "

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
param_file = ""
input_folder = ""


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


# Split the dataframe based on whether it is a [0->1] or [1->0] transitions
# and output to respective files.
# Timestamps file should have the header
def split_df_and_output(out_df, out_file_zero_to_one, out_file_one_to_zero,
                        out_file_zero_to_one_ts, out_file_one_to_zero_ts):
    if out_df.empty:
        return
    out_zero_to_one_df = pd.DataFrame(columns=out_col_names)
    out_one_to_zero_df = pd.DataFrame(columns=out_col_names)
    out_zero_to_one_df = out_df.loc[out_df[OUTPUT_COL3_FREEZE_TP]
                                    == ZERO_TO_ONE]
    out_zero_to_one_ts_df = out_zero_to_one_df.loc[:, OUTPUT_COL0_TS]
    out_one_to_zero_df = out_df.loc[out_df[OUTPUT_COL3_FREEZE_TP]
                                    == ONE_TO_ZERO]
    out_one_to_zero_ts_df = out_one_to_zero_df.loc[:, OUTPUT_COL0_TS]
    out_zero_to_one_df.to_csv(out_file_zero_to_one, index=False)
    out_zero_to_one_ts_df.to_csv(
        out_file_zero_to_one_ts, index=False, header=True)
    out_one_to_zero_df.to_csv(out_file_one_to_zero, index=False)
    out_one_to_zero_ts_df.to_csv(
        out_file_one_to_zero_ts, index=False, header=True)


# Format the output file names
def format_out_file_names(input_file, output_folder):
    global out_base
    global out_file_zero_to_one, out_file_zero_to_one_un
    global out_file_zero_to_one_ts, out_file_zero_to_one_un_ts
    global out_file_one_to_zero, out_file_one_to_zero_un
    global out_file_one_to_zero_ts, out_file_one_to_zero_un_ts

    input_file_name = os.path.basename(input_file)
    input_file_without_ext = os.path.splitext(input_file_name)[0]
    out_base = os.path.join(
        output_folder, input_file_without_ext + OUTPUT_BASE
    )
    out_file_zero_to_one = os.path.join(
        output_folder, input_file_without_ext + OUTPUT_ZERO_TO_ONE_CSV_NAME
    )
    out_file_zero_to_one_ts = os.path.join(
        output_folder, input_file_without_ext + OUTPUT_ZERO_TO_ONE_CSV_NAME_TS
    )
    out_file_zero_to_one_un = os.path.join(
        output_folder, input_file_without_ext + OUTPUT_ZERO_TO_ONE_UN_CSV_NAME
    )
    out_file_zero_to_one_un_ts = os.path.join(
        output_folder, input_file_without_ext + OUTPUT_ZERO_TO_ONE_UN_CSV_NAME_TS
    )
    out_file_one_to_zero = os.path.join(
        output_folder, input_file_without_ext + OUTPUT_ONE_TO_ZERO_CSV_NAME
    )
    out_file_one_to_zero_ts = os.path.join(
        output_folder, input_file_without_ext + OUTPUT_ONE_TO_ZERO_CSV_NAME_TS
    )
    out_file_one_to_zero_un = os.path.join(
        output_folder, input_file_without_ext + OUTPUT_ONE_TO_ZERO_UN_CSV_NAME
    )
    out_file_one_to_zero_un_ts = os.path.join(
        output_folder, input_file_without_ext + OUTPUT_ONE_TO_ZERO_UN_CSV_NAME_TS
    )

    print("\nInput file: ", os.path.basename(input_file_name))
    print("Output file:")
    print("\t[0->1]: ", os.path.basename(out_file_zero_to_one))
    print("\t[1->0]: ", os.path.basename(out_file_one_to_zero))
    print("\t[0->1] TimeStamps Only: ",
          os.path.basename(out_file_zero_to_one_ts))
    print("\t[1->0] TimeStamps Only: ",
          os.path.basename(out_file_one_to_zero_ts))
    print("\tunspecified [0->1]: ", os.path.basename(out_file_zero_to_one_un))
    print("\tunspeified [1->0]: ", os.path.basename(out_file_one_to_zero_un))
    print("\tunspecified [0->1] TimeStamps Only:: ",
          os.path.basename(out_file_zero_to_one_un_ts))
    print("\tunspeified [1->0] TimeStamps Only:: ",
          os.path.basename(out_file_one_to_zero_un_ts))

# Parse the input file
# returns bool, dataframe
# bool - True if parsing was successful, FALSE otherwise
# dataframe - Parsed dataframe


def parse_input_file_into_df(input_file):
    in_col_names = [INPUT_COL0, INPUT_COL1, INPUT_COL2]
    df = pd.read_csv(input_file, names=in_col_names,
                     skiprows=NUM_INITIAL_ROWS_TO_SKIP)

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


# Parse the paramter file
def parse_param_file():
    global param_file, param_min_time_duration
    global param_window_duration, param_start_timestamp_series
    global input_folder
    param_min_time_duration = 0
    param_window_duration = 0
    param_start_timestamp_series = pd.Series(dtype=np.float64)
    # Get the parameters folder, which is:
    # '<input folder>\parameters'
    param_file = os.path.join(input_folder, PARAMETERS_DIR_NAME)
    if os.path.isfile(param_file):
        param_df = pd.read_csv(
            param_file, names=param_col_names, header=None, skiprows=1)

        value = param_df[PARAM_TS_CRITERIA].iat[0]
        if not pd.isnull(value):
            param_min_time_duration = value
            print("Parameter - time duration: ",
                  param_min_time_duration)

        value = param_df[PARAM_TIME_WINDOW_DURATION].iat[0]
        if not pd.isnull(value):
            param_window_duration = value
            print("Parameter - window duration: ", param_window_duration)

        param_start_timestamp_series = param_df[PARAM_TIME_WINDOW_START_LIST]
        param_start_timestamp_series.sort_values(ascending=True)

    set_min_time_duration_box_value()
    set_time_window_duration_box_value()
    refresh_ts_list_box(param_start_timestamp_series)

# Print a dataframe with the message


def print_df(msg, df):
    if df.empty:
        return

    logging.debug(msg)
    logging.debug(df)


# Main logic routine to parse the input and spit out the output
def process_input_file(input_file, output_folder):
    global out_base
    global out_file_zero_to_one, out_file_zero_to_one_un
    global out_file_zero_to_one_ts, out_file_zero_to_one_un_ts
    global out_file_one_to_zero, out_file_one_to_zero_un
    global out_file_one_to_zero_ts, out_file_one_to_zero_un_ts
    global param_min_time_duration, param_window_duration, param_start_timestamp_series

    # Parse the input file
    success, df = parse_input_file_into_df(input_file)
    if not success:
        return

    # Parse the parameters file.
    parse_param_file()

    # Format the file names of the output files
    format_out_file_names(input_file, output_folder)

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
    out_df.to_csv(out_base, index=False)

    # Apply any minimum time duration criteria
    if param_min_time_duration > 0:
        out_df = out_df.loc[list(apply_duration_criteria(
            out_df.iloc[:, 0], param_min_time_duration))]
        print_df("After applying min time duration criteria", out_df)

    # Apply time window filter
    out_unspecified_df = pd.DataFrame()
    if not param_start_timestamp_series.empty:
        filter_list = list(apply_timewindow_filter(
            out_df.iloc[:, 0], param_start_timestamp_series, param_window_duration))
        out_unspecified_df = out_df.loc[~out_df.index.isin(filter_list)]
        out_df = out_df.loc[filter_list]
        print_df("After applying timestamp filter", out_df)
        print_df("unspecified entries after applying timestamp filter",
                 out_unspecified_df)

    split_df_and_output(out_df, out_file_zero_to_one, out_file_one_to_zero,
                        out_file_zero_to_one_ts, out_file_one_to_zero_ts)
    split_df_and_output(out_unspecified_df,
                        out_file_zero_to_one_un, out_file_one_to_zero_un,
                        out_file_zero_to_one_un_ts, out_file_one_to_zero_un_ts)

    return True


# Print help
def print_help():
    print("\nHelp/Usage:\n")
    print(
        "python freeze.py -i <input folder or .csv file> -d <output_directory> -v -h\n"
    )
    print("where:")
    print("-i (required): input folder or .csv file.")
    print("-d (optional): output folder.")
    print("-v (optional): run in verbose mode")
    print("-h (optional): print this help")
    print("\nExamples:\n")
    print("\nProcess input file:")
    print("\tpython freeze.py -i input.csv")
    print("\nProcess all the csv files from the input folder:")
    print("\tpython freeze.py -i c:\\data\\input")
    print(
        "\nProcess all the csv files from the input folder and use the output folder:"
    )
    print("\tpython freeze.py -i c:\\data\\input -d c:\\data\\output")
    print("\nNotes:")
    print("\tClose the output file prior to running.")
    sys.exit()


def main(argv, input_folder_or_file):
    global input_folder
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

    # get the input file if one is not provided.
    if input_folder_or_file == "":
        input_folder_or_file = input(
            "Provide an input folder or .csv file name: ")

    if os.path.isdir(input_folder_or_file):
        input_folder = input_folder_or_file
        search_path = input_folder_or_file + "\*.csv"
        for file in glob.glob(search_path):
            input_files.append(file)
    elif os.path.isfile(input_folder_or_file):
        input_folder = os.path.dirname(input_folder_or_file)
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

    print("\nInput folder: ", input_folder)
    print("Parameters file: ", param_file)
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

# ------------------------------------------------------------
# UI related stuff
# ------------------------------------------------------------


def line():
    Text(app, "------------------------------------------------------------------------------")


def select_input_folder():
    global input_folder
    input_dir_box.value = app.select_folder()
    output_folder_textbox.value = ""
    if not input_dir_box.value:
        return

    input_folder = input_dir_box.value
    parse_param_file()


def open_output_folder():
    global output_dir
    if not output_dir:
        app.warn(
            "Uh oh!", "Output folder dose not exist! Select input folder and process first.")
        return

    subprocess.Popen(f'explorer /select,{output_dir}')


def open_params_file():
    global param_file
    if not param_file:
        app.warn(
            "Uh oh!", "Parameters file does not exist. Please create a parmeters file called 'params.csv' in the <input folder>\parameters folder and try again!")
        return

    if not os.path.isfile(param_file):
        app.warn(
            "Uh oh!", "Parameters file " + param_file + " is not a file!")
        return

    os.startfile(param_file)
    parse_param_file()


def refresh_result_text_box(successfully_parsed_files, unsuccessfully_parsed_files):
    result_success_list_box.clear()
    for val in successfully_parsed_files:
        result_success_list_box.append(val)


def process():
    global output_dir
    if not input_dir_box.value:
        app.warn(
            "Uh oh!", "No input folder specified. Please select an input folder and run again!")
        return

    print("param_min_time_duration is : ", min_time_duration_box.value)
    output_dir, successfully_parsed_files, unsuccessfully_parsed_files = main(
        sys.argv[1:], input_dir_box.value)
    output_folder_textbox.value = ("Output folder: " + output_dir)
    refresh_result_text_box(successfully_parsed_files,
                            unsuccessfully_parsed_files)
    # result_window.show()


def refresh_ts_list_box(ts_series):
    ts_series_list_box.clear()
    for val in ts_series:
        ts_series_list_box.append(val)


def set_min_time_duration_box_value():
    global param_min_time_duration
    min_time_duration_box.value = (
        PARAM_UI_MIN_TIME_DURATION_CRITERIA_TEXT + str(param_min_time_duration))


def set_time_window_duration_box_value():
    global param_window_duration
    time_window_duration_box.value = (
        PARAM_UI_TIME_WINDOW_DURATION + str(param_window_duration))


# main entry point
if __name__ == "__main__":
    app = App("",  height=750, width=800)
    titlebox = TitleBox(app, "")
    title = Text(titlebox, text="Freeze Data Processing App",
                 size=14, font="Arial")
    line()
    Text(app, "Select Input Folder and then process", font="Verdana bold")
    line()
    input_folder_button = PushButton(
        app, command=select_input_folder, text="Input Folder")
    input_folder_button.tk.config(font=("Verdana bold", 16))
    input_dir_box = Text(app)
    line()
    min_time_duration_box = Text(app, text="")
    set_min_time_duration_box_value()
    time_window_duration_box = Text(app, text="")
    set_time_window_duration_box_value()
    Text(app, text=PARAM_UI_TIME_WINDOW_START_TIMES)
    ts_series_list_box = ListBox(
        app, param_start_timestamp_series, scrollbar=True)
    center_box = Box(app, layout="grid")
    open_params_button = PushButton(center_box, command=open_params_file,
                                    text="Open parameters file", grid=[0, 1])
    open_params_button.tk.config(font=("Verdana bold", 10))
    update_params_button = PushButton(center_box, text="Update parameters",
                                      command=parse_param_file, grid=[1, 1])
    update_params_button.tk.config(font=("Verdana bold", 10))
    line()
    process_button = PushButton(app, text="Process", command=process)
    process_button.tk.config(font=("Verdana bold", 10))
    line()
    output_folder_textbox = Text(app, text="", color="green")
    browse_output_folder_button = PushButton(
        app, command=open_output_folder, text="Browse output folder")
    browse_output_folder_button.tk.config(font=("Verdana bold", 10))
    line()
    result_window = Window(app, title="Result Window",
                           height=500, width=500, visible=False)
    result_text_box = Text(result_window, text="")
    result_success_list_box = ListBox(result_window, [], scrollbar=True)
    result_window.hide()
    app.display()
