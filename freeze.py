#!/usr/bin/python

import sys
import getopt
import logging
import os
import pandas as pd
from pandas.api.types import is_numeric_dtype
import glob
import guizero
from guizero import App, Box, Combo, ListBox, PushButton, Text, TextBox, TitleBox, Window
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
param_file = ""
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


def split_df_and_output(out_df, out_file_zero_to_one, out_file_one_to_zero,
                        out_file_zero_to_one_ts, out_file_one_to_zero_ts):
    """
    Split the dataframe based on whether it is a [0->1] or [1->0] transitions
    and output to respective files.
    Timestamps file should have the header
    """
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


def format_out_file_names(input_file, output_folder):
    """
    Format the output file names
    """
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


def parse_input_file_into_df(input_file):
    """
    Parse the input file
    returns bool, dataframe
    bool - True if parsing was successful; False otherwise
    dataframe - Parsed dataframe
    """
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


def parse_param_file():
    """
    Parse the paramter file
    """
    global param_file, param_min_time_duration
    global param_window_duration, param_start_timestamp_series
    global input_dir

    param_min_time_duration = 0
    param_window_duration = 0
    param_start_timestamp_series = pd.Series(dtype=np.float64)
    # Get the parameters folder, which is:
    # '<input folder>\parameters'
    param_file = os.path.join(input_dir, PARAMETERS_DIR_NAME)
    if os.path.isfile(param_file):
        param_df = pd.read_csv(
            param_file, names=param_col_names, header=None, skiprows=1)
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


def print_help():
    """
    Display help
    """
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

    # get the input file if one is not provided.
    if input_folder_or_file == "":
        input_folder_or_file = input(
            "Provide an input folder or .csv file name: ")

    if os.path.isdir(input_folder_or_file):
        input_dir = input_folder_or_file
        search_path = input_folder_or_file + "\*.csv"
        for file in glob.glob(search_path):
            input_files.append(file)
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


"""
------------------------------------------------------------
                UI related stuff
------------------------------------------------------------
"""


def line():
    Text(app, "------------------------------------------------------------------------------------------------------")


def select_input_folder():
    global input_dir, param_name_list

    input_dir_temp = app.select_folder()
    if not input_dir_temp:
        return

    input_dir = input_dir_temp
    parse_param_file()
    parse_param_folder()
    refresh_param_names_combo_box(param_name_list)


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


def refresh_result_text_box(successfully_parsed_files, unsuccessfully_parsed_files):
    result_success_list_box.clear()
    for val in successfully_parsed_files:
        result_success_list_box.append(val)


def process():
    global output_dir
    if not input_dir:
        app.warn(
            "Uh oh!", "No input folder specified. Please select an input folder and run again!")
        return

    print("param_min_time_duration is : ", min_time_duration_box.value)
    output_dir, successfully_parsed_files, unsuccessfully_parsed_files = main(
        sys.argv[1:], input_dir)
    refresh_result_text_box(successfully_parsed_files,
                            unsuccessfully_parsed_files)
    # result_window.show()


def select_param(selected_param_value):
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


if __name__ == "__main__":
    """
    Main entry point
    """
    # Main app
    app = App("",  height=750, width=800)

    # Title box
    titlebox = TitleBox(app, "")
    titlebox.bg = "white"
    title = Text(app, text="Freeze Data Processing App",
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
    # param_names_combo_box.hide()
    cnt += 1

    # Boxes related to showing parameters
    min_time_duration_label_box = Text(
        param_box, text=PARAM_UI_MIN_TIME_DURATION_CRITERIA_TEXT, grid=[0, cnt], align="left")
    min_time_duration_box = TextBox(
        param_box, text="", grid=[1, cnt], align="left")
    set_min_time_duration_box_value()
    cnt += 1

    time_window_duration_label_box = Text(
        param_box, text=PARAM_UI_TIME_WINDOW_DURATION_TEXT, grid=[0, cnt], align="left")
    time_window_duration_box = TextBox(
        param_box, text="", grid=[1, cnt], align="left")
    set_time_window_duration_box_value()
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
    update_params_button = PushButton(center_box, text="Update parameters",
                                      command=parse_param_file, grid=[1, 1], width=17, align="left")
    update_params_button.tk.config(font=("Verdana bold", 10))
    line()

    # Process input button
    process_button = PushButton(app, text="Process", command=process, width=26)
    process_button.tk.config(font=("Verdana bold", 14))
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
