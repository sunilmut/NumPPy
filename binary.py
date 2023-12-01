#!/usr/bin/python

import sys
import getopt
import logging
import os
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_integer_dtype
import glob
from guizero import App, Box, CheckBox, Combo, ListBox, PushButton, Text, TextBox, TitleBox, Window
import subprocess
import numpy as np
import csv
from csv import reader
import unittest
import common
from parameter import *

# Constants:
ZERO_TO_ONE = "0 to 1"
ONE_TO_ZERO = "1 to 0"

# output columns
OUTPUT_COL0_TS = "timestamps"
OUTPUT_COL1_MI = "Motion Index"
OUTPUT_COL2_MI_AVG = "Avg of MI"
OUTPUT_COL3_FREEZE_TP = "Freezing TurnPoints"

# output directory and file names
OUTPUT_DIR_NAME = "_output"
TIME_DURATION_PARAMETER_FILE = "min_time.txt"
OUTPUT_BASE = "_base.csv"
MIN_DURATION = "_min_duration.csv"
TW_FILTER = "_tw_filter.csv"
UNDERSCORE = "_"
EMPTY = "_EMPTY"

# output file name formats:
OUTPUT_NO_PARAMETERS = "_Not"
OUTPUT_ZERO_TO_ONE_CSV_NAME = "01"
OUTPUT_ONE_TO_ZERO_CSV_NAME = "10"
OUTPUT_LOG_FILE = "output.txt"

# UI related constants
INPUT_FOLDER_NAME_BOX_MAX_WIDTH = 26
PARAM_UI_TIME_WINDOW_START_TIMES = "Time window start (secs):"
PARAM_UI_MIN_TIME_DURATION_CRITERIA_TEXT = "Min time duration criteria (secs):"
PARAM_UI_MIN_BEFORE_TIME_DURATION_CRITERIA_TEXT = "\t\t    before: "
PARAM_UI_MIN_AFTER_TIME_DURATION_CRITERIA_TEXT = "after: "
PARAM_UI_TIME_WINDOW_DURATION_TEXT = "Time window duration (secs): "

# globals
logger = None
r_log_box = None
output_dir = None
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

# Arrays to store the parameter names and its value as a dataframe
# There is a dataframe value for each parameter and the indexes
# for these two arrays should be kept in sync.
# Parameter names
param_name_list = []
# Parameter values as dataframe. There is one dataframe for each parameter
param_df_list = []
param_file_exists = False
parameter_obj = Parameters()

param_min_time_duration_before = 0
param_min_time_duration_after = 0
param_window_duration = 0

# When set to True, only the currently selected parameter is processed instead
# of all the parameters. Default is to process all parameters.
only_process_cur_param = False

# When set to True, create output folder with the csv file name.
create_output_folder_with_file_name = True

# When set to True, add the csv file name to the output file names.
add_csv_file_name_to_output = False

files_without_timeshift = []


def apply_duration_criteria(ts_series, param_min_time_duration_before, param_min_time_duration_after):
    it = ts_series.iteritems()
    prev_ts = 0
    for i, (idx, ts) in enumerate(it):
        # Query the timestamp of the next element in the series.
        # Default behavior is to drop the last element.
        if i == ts_series.size - 1:
            break
        next_ts = ts_series.iat[i + 1]
        if ((ts - prev_ts >= param_min_time_duration_before) and (next_ts - ts >= param_min_time_duration_after)):
            yield idx
        prev_ts = ts


def apply_timewindow_filter(ts_series, timstamp_filter_series, duration):
    it = ts_series.iteritems()
    for idx, val in it:
        for filter_idx, filter_val in timstamp_filter_series.items():
            if val < filter_val:
                break
            if val >= filter_val and val <= filter_val + duration:
                yield idx
                break


def apply_special_emtpy_ts_logic(file_name_with_path, df):
    """
    This function will generate a new name for the file if the supplied
    dataframe is empty
    """

    if df.empty:
        dir_namm = os.path.dirname(file_name_with_path)
        file_name_without_ext, ext = os.path.splitext(
            os.path.basename(file_name_with_path))

        return os.path.join(dir_namm, file_name_without_ext + EMPTY + ext)

    return file_name_with_path


def split_df_and_output(out_df, timeshift_val, out_file_zero_to_one, out_file_one_to_zero,
                        out_file_zero_to_one_ts, out_file_one_to_zero_ts):
    """
    Split the dataframe based on whether it is a [0->1] or [1->0] transitions
    and output to respective files.
    Timestamps file should have the header
    """

    global logger

    # Create a copy of the dataframe so that the original is not affected
    output_df = out_df[:]

    output_df[OUTPUT_COL0_TS] = output_df[OUTPUT_COL0_TS] + timeshift_val
    # Round up to two decimals
    output_df[OUTPUT_COL0_TS] = output_df[OUTPUT_COL0_TS].round(decimals=2)
    common.logger.debug("Removing all entries with timestamp < 10 secs")
    # Remove all entries with timeseconds less than 10 seconds
    output_df = output_df[output_df[OUTPUT_COL0_TS] > 10]

    out_zero_to_one_df = pd.DataFrame(columns=out_col_names)
    out_one_to_zero_df = pd.DataFrame(columns=out_col_names)
    out_zero_to_one_df = output_df.loc[output_df[OUTPUT_COL3_FREEZE_TP]
                                       == ZERO_TO_ONE]
    out_zero_to_one_ts_df = out_zero_to_one_df.loc[:, OUTPUT_COL0_TS]
    out_one_to_zero_df = output_df.loc[output_df[OUTPUT_COL3_FREEZE_TP]
                                       == ONE_TO_ZERO]
    out_one_to_zero_ts_df = out_one_to_zero_df.loc[:, OUTPUT_COL0_TS]
    out_zero_to_one_df.to_csv(out_file_zero_to_one, index=False)
    out_file_zero_to_one_ts = apply_special_emtpy_ts_logic(
        out_file_zero_to_one_ts, out_zero_to_one_ts_df)
    out_zero_to_one_ts_df.to_csv(
        out_file_zero_to_one_ts, index=False, header=True)
    out_one_to_zero_df.to_csv(out_file_one_to_zero, index=False)
    out_file_one_to_zero_ts = apply_special_emtpy_ts_logic(
        out_file_one_to_zero_ts, out_one_to_zero_ts_df)
    out_one_to_zero_ts_df.to_csv(
        out_file_one_to_zero_ts, index=False, header=True)


def out_base(input_file, output_folder):
    input_file_without_ext = os.path.splitext(os.path.basename(input_file))[0]
    return os.path.join(
        output_folder, input_file_without_ext + OUTPUT_BASE
    )


def out_min_duration_file(input_file, output_folder):
    input_file_without_ext = os.path.splitext(os.path.basename(input_file))[0]
    return os.path.join(
        output_folder, input_file_without_ext + MIN_DURATION
    )


def out_tw_filter_file(input_file, param_name, output_folder):
    input_file_without_ext = os.path.splitext(os.path.basename(input_file))[0]
    return os.path.join(
        output_folder, input_file_without_ext + UNDERSCORE + param_name + TW_FILTER
    )


def format_out_name_with_param_val(param_min_time_duration_before,
                                   param_min_time_duration_after):
    global param_file_exists

    if param_file_exists == False or (param_min_time_duration_before == 0 and param_min_time_duration_after == 0):
        return ""
    else:
        return UNDERSCORE + str(int(param_min_time_duration_before)) + "-" + str(int(param_min_time_duration_after)) + "s"


def format_out_nop_file_name(input_file, param_names, param_min_time_duration_before,
                             param_min_time_duration_after, output_folder):
    """
    Format the 'no parameter' output file name
    """
    global out_file_zero_to_one_un, out_file_zero_to_one_un_ts
    global out_file_one_to_zero_un, out_file_one_to_zero_un_ts
    global logger

    output_no_parameter = ""
    if param_names:
        output_no_parameter = OUTPUT_NO_PARAMETERS

    input_file_without_ext = os.path.splitext(os.path.basename(input_file))[0]
    out_file_zero_to_one_un = os.path.join(
        output_folder, input_file_without_ext + UNDERSCORE +
        OUTPUT_ZERO_TO_ONE_CSV_NAME + output_no_parameter + param_names + common.CSV_EXT
    )
    min_time_before_after = format_out_name_with_param_val(
        param_min_time_duration_before, param_min_time_duration_after)

    if add_csv_file_name_to_output:
        out_file_zero_to_one_un_ts = os.path.join(
            output_folder,  OUTPUT_ZERO_TO_ONE_CSV_NAME + UNDERSCORE + input_file_without_ext +
            min_time_before_after + output_no_parameter + param_names + common.CSV_EXT
        )
    else:
        out_file_zero_to_one_un_ts = os.path.join(
            output_folder,  OUTPUT_ZERO_TO_ONE_CSV_NAME + min_time_before_after +
            output_no_parameter + param_names + common.CSV_EXT
        )

    out_file_one_to_zero_un = os.path.join(
        output_folder, input_file_without_ext + UNDERSCORE +
        OUTPUT_ONE_TO_ZERO_CSV_NAME + output_no_parameter + param_names + common.CSV_EXT
    )

    if add_csv_file_name_to_output:
        out_file_one_to_zero_un_ts = os.path.join(
            output_folder, OUTPUT_ONE_TO_ZERO_CSV_NAME + UNDERSCORE + input_file_without_ext +
            min_time_before_after + output_no_parameter + param_names + common.CSV_EXT
        )
    else:
        out_file_one_to_zero_un_ts = os.path.join(
            output_folder, OUTPUT_ONE_TO_ZERO_CSV_NAME + min_time_before_after +
            output_no_parameter + param_names + common.CSV_EXT
        )

    common.logger.debug("\tOutput files:")
    common.logger.debug("\t\tNOp [0->1]: %s",
                 os.path.basename(out_file_zero_to_one_un))
    common.logger.debug("\t\tNOp [1->0]: %s",
                 os.path.basename(out_file_one_to_zero_un))
    common.logger.debug("\t\tNOp [0->1] TimeStamps Only: %s",
                 os.path.basename(out_file_zero_to_one_un_ts))
    common.logger.debug("\t\tNOp [1->0] TimeStamps Only: %s",
                 os.path.basename(out_file_one_to_zero_un_ts))


def format_out_file_names(input_file, param_name,  param_min_time_duration_before,
                          param_min_time_duration_after, output_folder):
    """
    Format the output file names
    """
    global out_file_zero_to_one, out_file_zero_to_one_ts
    global out_file_one_to_zero, out_file_one_to_zero_ts
    global logger, add_csv_file_name_to_output

    param_ext = ""
    if param_name:
        param_ext = "_" + param_name

    input_file_without_ext = os.path.splitext(os.path.basename(input_file))[0]
    out_file_zero_to_one = os.path.join(
        output_folder, input_file_without_ext + UNDERSCORE +
        OUTPUT_ZERO_TO_ONE_CSV_NAME + param_ext + common.CSV_EXT
    )
    min_time_before_after = format_out_name_with_param_val(
        param_min_time_duration_before, param_min_time_duration_after)
    if add_csv_file_name_to_output:
        out_file_zero_to_one_ts = os.path.join(
            output_folder, OUTPUT_ZERO_TO_ONE_CSV_NAME + UNDERSCORE + input_file_without_ext +
            min_time_before_after + param_ext + common.CSV_EXT
        )
    else:
        out_file_zero_to_one_ts = os.path.join(
            output_folder, OUTPUT_ZERO_TO_ONE_CSV_NAME +
            min_time_before_after + param_ext + common.CSV_EXT
        )

    out_file_one_to_zero = os.path.join(
        output_folder, input_file_without_ext + UNDERSCORE +
        OUTPUT_ONE_TO_ZERO_CSV_NAME + param_ext + common.CSV_EXT
    )

    if add_csv_file_name_to_output:
        out_file_one_to_zero_ts = os.path.join(
            output_folder, OUTPUT_ONE_TO_ZERO_CSV_NAME + UNDERSCORE + input_file_without_ext +
            min_time_before_after + param_ext + common.CSV_EXT
        )
    else:
        out_file_one_to_zero_ts = os.path.join(
            output_folder, OUTPUT_ONE_TO_ZERO_CSV_NAME +
            min_time_before_after + param_ext + common.CSV_EXT
        )

    common.logger.debug("\tOutput files:")
    common.logger.debug("\t\t[0->1]: %s", os.path.basename(out_file_zero_to_one))
    common.logger.debug("\t\t[1->0]: %s", os.path.basename(out_file_one_to_zero))
    common.logger.debug("\t\t[0->1] TimeStamps Only: %s",
                 os.path.basename(out_file_zero_to_one_ts))
    common.logger.debug("\t\t[1->0] TimeStamps Only: %s",
                 os.path.basename(out_file_one_to_zero_ts))


def parse_param_folder():
    """
    Parse parameter folder and create a list of parameter dataframe(s)
    out of it.
    """
    global param_name_list, param_df_list

    input_dir = common.get_input_dir()

    try:
        parameter_obj.parse(input_dir)
    except ValueError as e:
        common.logger.warning(e)

    param_name_list = parameter_obj.get_param_name_list()
    param_df_list = parameter_obj.get_param_df()


def get_parameter_min_t_file():
    input_dir = common.get_input_dir()
    min_t_file = os.path.join(
        input_dir, Parameters.PARAMETERS_DIR_NAME, TIME_DURATION_PARAMETER_FILE)

    return min_t_file


def get_param_min_time_duration():
    """
    Returns the value of the min parameter time duration value.
    """
    global logger, param_min_time_duration_before
    global param_min_time_duration_after, param_file_exists

    t_duration_before = param_min_time_duration_before
    t_duration_after = param_min_time_duration_after
    itr = 0
    param_min_t_file = get_parameter_min_t_file()
    common.logger.debug("opening min time file %s", param_min_t_file)
    try:
        with open(param_min_t_file) as min_t_file:
            param_file_exists = True
            while True:
                line = min_t_file.readline().rstrip()
                if not line:
                    break
                common.logger.debug("line %s", line)
                try:
                    t_duration = float(line)
                    if itr == 0:
                        t_duration_before = t_duration
                        itr += 1
                    else:
                        t_duration_after = t_duration
                        break
                except ValueError:
                    common.logger.error(
                        "Min time duration(%s) from file(%s) cannot be converted to a "
                        "number. Using default of %d", line, param_min_t_file, t_duration)
            min_t_file.close()
    except IOError:
        param_file_exists = False
        common.logger.debug(
            "Min time duration file(%s) does not exist.", param_min_t_file)
        pass

    #common.logger.debug("min t before %d & after %d", t_duration_before, t_duration_after)
    return t_duration_before, t_duration_after


def parse_param_df(df):
    value = df[Parameters.PARAM_TIME_WINDOW_DURATION].iat[0]
    w_duration = 0
    if not pd.isnull(value):
        w_duration = value

    ts_series = df[Parameters.PARAM_TIME_WINDOW_START_LIST]
    ts_series.sort_values(ascending=True)

    return w_duration, ts_series


def reset_parameters():
    global param_min_time_duration_before, param_window_duration
    global param_min_time_duration_after

    param_min_time_duration_before = param_min_time_duration_before
    param_window_duration = param_min_time_duration_after


def reset_all_parameters():
    global param_name_list, param_df_list

    param_name_list.clear()
    param_df_list.clear()
    reset_parameters()


def parse_cur_param_file(parameter_obj):
    """
    Parse the paramter file
    """

    update_min_t_in_file(min_time_duration_before_box.value,
                         min_time_duration_after_box.value)

    currently_selected_param = parameter_obj.get_currently_selected_param()
    if not currently_selected_param:
        return

    try:
        parameter_obj.parse(common.get_input_dir())
        parameter_obj.set_currently_selected_param(currently_selected_param)
        param_window_duration, param_start_timestamp_series = parameter_obj.get_param_values(currently_selected_param)
    except ValueError as e:
        common.logger.error(e)

    refresh_param_values_ui(param_window_duration, param_start_timestamp_series)


def process_param(param_idx, out_df, nop_df):
    """
    Processes the dataframe for the parameter name specified using
    the parameter name.
    Returns:
    - dataframe after applying the time window criteria
    - 'not in parameter' dataframe - i.e. everything in the input dataframe after removing
      entries that were selected by the parameter criterias
      p.s - This parameter allows continuation from a previous 'nop_df'
    """
    global param_df_list

    # Make a copy by value
    temp_out_df = out_df[:]
    param_window_duration, param_start_timestamp_series = parse_param_df(
        param_df_list[param_idx])

    # Apply time window filter
    if not param_start_timestamp_series.empty:
        filter_list = list(apply_timewindow_filter(
            temp_out_df.iloc[:, 0], param_start_timestamp_series, param_window_duration))
        temp_out_df = temp_out_df.loc[filter_list]

    # Build a consolidated (for each parameter) 'not in parameter' dataframe
    # after starting from original set and by removing entries that are in the parameter.
    # This is done by doing an right outer join of the two dataframes where the right
    # dataframe is 'not in parameter' dataframe.
    nop_df = pd.merge(temp_out_df, nop_df, how='outer', indicator=True).query(
        "_merge == 'right_only'").drop('_merge', axis=1).reset_index(drop=True)

    return temp_out_df, nop_df


def process_input_df(input_df):
    """
    Process an input dataframe and return an output base dataframe (i.e. without
    any of the criterias or parameters applied)
    """
    global logger

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
    for (idx, row) in input_df.iterrows():
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
                common.logger.error("Current index: %s", str(idx + 4))
                common.logger.error("Row value: %s", row.to_string())
                common.logger.error("Previous index: %s", str(prev_idx + 4))
                return False, None

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

    return True, out_df


def apply_min_time_duration_criteria(min_t_before, min_t_after, df):
    """
    This will apply minimum time duration criteria on the provided
    dataframe and return the dataframe.
    """

    if min_t_before > 0 or min_t_after > 0:
        df = df.loc[list(apply_duration_criteria(
            df.iloc[:, 0], min_t_before, min_t_after))]

    return df


def process_input_file(input_file, output_folder):
    """
    Main logic routine to parse the input and spit out the output
    """
    global out_file_zero_to_one, out_file_zero_to_one_un
    global out_file_zero_to_one_ts, out_file_zero_to_one_un_ts
    global out_file_one_to_zero, out_file_one_to_zero_un
    global out_file_one_to_zero_ts, out_file_one_to_zero_un_ts
    global param_min_time_duration_before, param_window_duration
    global param_min_time_duration_after, files_without_timeshift
    global param_file_exists, parameter_obj

    common.logger.debug("Processing input file: %s", os.path.basename(input_file))
    timeshift_val, num_rows_processed = common.get_timeshift_from_input_file(
        input_file)

    if timeshift_val:
        common.logger.debug("\tUsing timeshift value of: %s", str(timeshift_val))
    else:
        if timeshift_val is None:
            timeshift_val = 0
            common.logger.warning("\tNo timeshift value specified")
        else:
            common.logger.warning("\tIncorrect timeshift value of zero specified")
        files_without_timeshift.append(input_file)

    # Parse the input file
    success, df = common.parse_input_file_into_df(
        input_file, common.NUM_INITIAL_ROWS_TO_SKIP + num_rows_processed)
    if not success:
        return False

    # Parse all the parameter files.
    reset_all_parameters()
    parse_param_folder()

    # Get a base output dataframe without any criterias applied
    result, out_df = process_input_df(df)
    if result == False or out_df.empty:
        return False

    out_base_file = out_base(input_file, output_folder)
    common.logger.debug(
        "\tAfter initial parsing (without any criteria or filter): %s",
        os.path.basename(out_base_file))
    out_df.to_csv(out_base_file, index=False)

    param_min_time_duration_before, param_min_time_duration_after = get_param_min_time_duration()
    if param_file_exists:
        common.logger.debug("\tUsing min time duration (secs): %s",
                     str(param_min_time_duration_before))
        # Apply any minimum time duration criteria
        out_df = apply_min_time_duration_criteria(
            param_min_time_duration_before, param_min_time_duration_after, out_df)

    min_duration_file = out_min_duration_file(input_file, output_folder)
    common.logger.debug("\tAfter applying min time duration "
                 "criteria: %s", os.path.basename(min_duration_file))
    out_df.to_csv(min_duration_file, index=False)

    # 'nop' -> not in parameter. This is the left over of the original data
    # after all the parameters have been processed. 'no' starts with
    # the original set (after the min time duration) and as each parameter
    # gets process the set gets subtracted by that.
    # Make a copy of out dataframe 'by value' and not by reference.
    nop_df = out_df[:]
    params_name = ""
    cur_selected_param = parameter_obj.get_currently_selected_param()

    # Iterate through the parameters and apply each one of them
    for idx, param_name in enumerate(param_name_list):
        common.logger.debug("\tProcessing parameter: %s", param_name)
        if only_process_cur_param and param_name != cur_selected_param:
            continue

        params_name += UNDERSCORE + param_name

        # process the dataframe for the parameter with the given index.
        temp_out_df, nop_df = process_param(idx, out_df, nop_df)

        # Format the file names of the output files using the parameter name
        format_out_file_names(input_file, param_name,  param_min_time_duration_before,
                              param_min_time_duration_after, output_folder)
        tw_filter_file = out_tw_filter_file(
            input_file, param_name, output_folder)
        common.logger.debug(
            "\tAfter applying time window duration filter: %s",
            os.path.basename(tw_filter_file))
        temp_out_df.to_csv(tw_filter_file, index=False)

        split_df_and_output(temp_out_df, timeshift_val, out_file_zero_to_one, out_file_one_to_zero,
                            out_file_zero_to_one_ts, out_file_one_to_zero_ts)

    format_out_nop_file_name(input_file, params_name, param_min_time_duration_before,
                             param_min_time_duration_after, output_folder)
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
        "python binary.py -i <input folder or .csv file> -o <output_directory> -v -h -c\n"
    )
    print("where:")
    print("-i (required): input folder or .csv file.")
    print("-o (optional): output folder.")
    print("-v (optional): run in verbose mode")
    print("-h (optional): print this help")
    print("-c (optional): run in console (no UI) mode")
    print("\nExamples:\n")
    print("\nProcess input file:")
    print("\tpython binary.py -i input.csv")
    print("\nProcess all the csv files from the input folder:")
    print("\tpython binary.py -i c:\\data\\input")
    print(
        "\nProcess all the csv files from the input folder and use the output folder:"
    )
    print("\tpython binary.py -i c:\\data\\input -d c:\\data\\output")
    print("\nNotes:")
    print("\tClose the output file prior to running.")
    sys.exit()


def get_input_files(input_dir):
    input_files = []

    # Normalize the path to deal with backslash/frontslash
    input_folder_or_file = os.path.normpath(input_dir)
    search_path = os.path.join(input_folder_or_file, "*.csv")
    for file in glob.glob(search_path):
        input_files.append(file)

    return input_files


class loghandler(logging.StreamHandler):
    """
    Custom logging handler
    """

    def emit(self, record):
        """
        Writes the message to the output file (or the default logger stream),
        stdout and the UI result text box
        """
        try:
            # TODO: Have different log format for input depending on whether
            # it is the file output or anything else. And, also based on the
            # log level. Ex: error logs can be prepend with some string like
            # "ERROR:: "
            msg = self.format(record)
            if r_log_box is not None:
                r_log_box.value += msg
            print(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def main(input_folder_or_file, separate_files, output_folder):
    global output_dir, create_output_folder_with_file_name

    input_dir = ""
    input_files = []
    output_folder = ""

    # strip the quotes at the start and end, else
    # paths with white spaces won't work.
    input_folder_or_file = input_folder_or_file.strip('"')

    # get the input file if one is not provided.
    if not input_folder_or_file:
        input_folder_or_file = input(
            "Provide an input folder or .csv file name: ")

    input_folder_or_file = os.path.normpath(input_folder_or_file)
    if os.path.isdir(input_folder_or_file):
        input_dir = input_folder_or_file
        input_files = get_input_files(input_dir)
    elif os.path.isfile(input_folder_or_file):
        input_dir = os.path.dirname(input_folder_or_file)
        input_files.append(input_folder_or_file)
    else:
        common.logger.error("The input path is not a valid directory or file: %s",
                     input_folder_or_file)
        print_help()

    # TODO: Use the common method.
    # If an output folder is specified, use it.
    # Else, output folder is '<parent of input file or folder>\output', create it
    if not output_folder:
        output_folder = os.path.dirname(input_folder_or_file)
        base_name = os.path.basename(input_folder_or_file)
        if separate_files:
            output_folder = os.path.join(output_folder, base_name)
        else:
            output_folder = os.path.join(
                output_folder, base_name + OUTPUT_DIR_NAME)
            if not os.path.isdir(output_folder):
                os.mkdir(output_folder)

    common.set_input_dir(input_dir)
    common.logger.debug("Input folder: %s", os.path.normpath(input_dir))
    output_dir = os.path.normpath(output_folder)
    successfully_parsed_files = []
    unsuccessfully_parsed_files = []
    for input_file in input_files:
        # Create a different output folder for each input file
        if separate_files or create_output_folder_with_file_name:
            output_folder = os.path.join(
                output_dir, os.path.splitext(os.path.basename(input_file))[0])
        common.logger.debug("Output folder: %s", output_folder)
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        if separate_files:
            parsed = separate_input_file(input_file, output_folder)
        else:
            parsed = process_input_file(input_file, output_folder)

        if parsed:
            successfully_parsed_files.append(input_file)
        else:
            unsuccessfully_parsed_files.append(input_file)

    return successfully_parsed_files, unsuccessfully_parsed_files


def separate_input_file(input_file, output_folder):
    """
    Main logic routine to separate input file into multiple files.
    """

    global files_without_timeshift
    timeshift_val, num_rows_processed = common.get_timeshift_from_input_file(
        input_file)

    if timeshift_val:
        common.logger.debug("\tUsing timeshift value of: %s", str(timeshift_val))
    else:
        if timeshift_val is None:
            timeshift_val = 0
            common.logger.warning("\tNo timeshift value specified")
        else:
            common.logger.warning("\tIncorrect timeshift value of zero specified")
        files_without_timeshift.append(input_file)

    # Parse the input file
    df = pd.read_csv(input_file, header=0, skiprows=num_rows_processed)
    columns = df.columns.str.lower().to_list()
    try:
        time_col_index = columns.index("time")
    except ValueError:
        common.logger.warning("Time column is not present in the file. Skipping file")
        return False

    for i, col in enumerate(df):
        if i == time_col_index:
            continue

        # Binary whole numbers only
        if (
            is_integer_dtype(df[col])
            and df[col].min() == 0
            and df[col].max() == 1
        ):
            out_df_frame = pd.DataFrame(
                [*zip([common.TIMESHIFT_HEADER_ALT, common.INPUT_COL0_TS],
                      [int(timeshift_val), common.INPUT_COL1_MI],
                      [np.nan, common.INPUT_COL2_FREEZE])])
            output_file_name = os.path.join(
                output_folder, col.replace(" ", "_") + common.CSV_EXT)
            out_df_freeze = df[col]
            # Empty motion index columns
            out_mi = pd.DataFrame(index=range(
                out_df_freeze.size), columns=range(1))
            out_df_time = df.iloc[:, time_col_index]
            out_df = pd.concat([out_df_time, out_mi, out_df_freeze],
                               axis=1, ignore_index=True, sort=False)
            out_df = pd.concat([out_df_frame, out_df],
                               ignore_index=True, sort=False)
            out_df.to_csv(output_file_name, index=False, header=False)
            # print(out_df)

    return True


"""
------------------------------------------------------------
                UI related stuff
------------------------------------------------------------
"""


def line():
    """
    Line for the main app
    """
    Text(app, "------------------------------------------------------------------------------------------------------")


def line_r(rwin):
    """
    Line for results window
    """
    Text(rwin, "-------------------------------------------------------------------------------------")

def select_input_dir(parameter_obj):
    global param_min_time_duration_before, param_min_time_duration_after

    input_dir = common.select_input_dir(app)
    try:
        parameter_obj.parse(input_dir)
    except ValueError as e:
        common.logger.warning(e)

    input_dir_text_box.value = os.path.basename(input_dir)
    input_dir_text_box.width = min(
        len(input_dir_text_box.value), INPUT_FOLDER_NAME_BOX_MAX_WIDTH)

    # Reset all current values of the parameters and refresh the parameter
    # UI section with the reset values (this will ensure the UI will show
    # the default values even in cases when there are no parameters specified
    # in the input folder).
    param_window_duration, param_start_timestamp_series = parameter_obj.get_default_parameter_values()
    refresh_param_values_ui(param_window_duration, param_start_timestamp_series)
    param_min_time_duration_before, param_min_time_duration_after = get_param_min_time_duration()
    param_name_list = parameter_obj.get_param_name_list()
    refresh_param_names_combo_box(parameter_obj)
    if len(param_name_list):
        param_window_duration, param_start_timestamp_series = parameter_obj.get_param_values(
            parameter_obj.get_currently_selected_param())
        refresh_param_values_ui(param_window_duration, param_start_timestamp_series)

def open_output_folder():
    global output_dir

    if not output_dir:
        app.warn(
            "Uh oh!", "Output folder dose not exist! Select input folder and process first.")
        return

    # Normalize the path to deal with backslash/frontslash
    norm_path = os.path.normpath(output_dir)
    if sys.platform == "win32":
        subprocess.Popen(f'explorer /open,{norm_path}')
    elif sys.platform == "darwin":
        subprocess.Popen(["open", norm_path])
    else:
        subprocess.Popen(["xdg-open", norm_path])


def open_params_file(parameter_obj):
    cur_selected_param = parameter_obj.get_currently_selected_param()
    update_min_t_in_file(min_time_duration_before_box.value,
                         min_time_duration_after_box.value)
    if not cur_selected_param:
        return

    param_file = parameter_obj.get_param_file_from_name(
        parameter_obj.get_currently_selected_param())
    if not os.path.isfile(param_file):
        app.warn(
            "Uh oh!", "Parameters file " + param_file + " is not a file!")
        return

    common.open_file(param_file)


def reset_result_box():
    result_success_list_box.clear()
    result_unsuccess_list_box.clear()
    result_withoutshift_list_box.clear()
    files_without_timeshift.clear()
    r_log_box.clear()


def refresh_result_text_box(successfully_parsed_files, unsuccessfully_parsed_files):
    """
    This function will refresh the result window with the updated results.
    """
    global files_without_timeshift

    total_files_processed = len(
        successfully_parsed_files) + len(unsuccessfully_parsed_files)
    result_box_str = "Total input files processed: " + \
        str(total_files_processed) + "\n"
    result_box_str += "Number of files successfully processed: " + \
        str(len(successfully_parsed_files)) + "\n"
    result_box_str += "Number of files failed to process: " + \
        str(len(unsuccessfully_parsed_files)) + "\n"
    result_box_str += "Number of files without shift values: " + \
        str(len(files_without_timeshift))
    result_text_box.value = result_box_str

    for val in successfully_parsed_files:
        result_success_list_box.append(os.path.basename(val))

    for val in unsuccessfully_parsed_files:
        result_unsuccess_list_box.append(os.path.basename(val))

    for val in files_without_timeshift:
        result_withoutshift_list_box.append(os.path.basename(val))


def update_min_t_in_file(min_t_before_val, min_t_after_val):
    """
    Will read the min time duration value(s) from the UI and update the
    min time duration value(s), both the global one and the one in the
    file.
    """
    global param_min_time_duration_before, param_min_time_duration_after, logger

    param_min_t_file = get_parameter_min_t_file()

    try:
        # "w+" will create the file if not exist.
        with open(param_min_t_file, "w+") as min_t_file:
            min_t_file.write(min_t_before_val)
            min_t_file.write("\n")
            min_t_file.write(min_t_after_val)
            min_t_file.close()
            param_min_time_duration_before = min_t_before_val
            param_min_time_duration_after = min_t_after_val
    except IOError:
        common.logger.error(
            "Min time duration file(%s) cannot be created or written to.", param_min_t_file)
        pass


def process():
    input_dir = common.get_input_dir()
    if not input_dir:
        app.warn(
            "Uh oh!", "No input folder specified. Please select an input folder and run again!")
        return

    # Reset the result box before processing so that the new values of the
    # processing results can be shown.
    reset_result_box()

    # Update the min time duration value in the file with the value
    # from the UI so that it sticks.
    # p.s: The main process loop will read this value from the file. So
    #      the update should be done prior to processing the files.
    update_min_t_in_file(min_time_duration_before_box.value,
                         min_time_duration_after_box.value)
    successfully_parsed_files, unsuccessfully_parsed_files = main(
        input_dir, False, None)
    refresh_result_text_box(successfully_parsed_files,
                            unsuccessfully_parsed_files)

    rwin.show()


def process_separate_input():
    input_dir = common.get_input_dir()
    if not input_dir:
        app.warn(
            "Uh oh!", "No input folder specified. Please select an input folder and run again!")
        return

    # Reset the result box before processing so that the new values of the
    # processing results can be shown.
    reset_result_box()
    successfully_parsed_files, unsuccessfully_parsed_files = main(
        input_dir, True, None)
    refresh_result_text_box(successfully_parsed_files,
                            unsuccessfully_parsed_files)

    rwin.show()


def select_param(selected_param_value):
    """
    This method is called when the user selects a parameter from the parameter
    name drop down box.
    """
    global param_min_time_duration_before, parameter_obj
    global param_min_time_duration_after, param_window_duration

    # First thing is to apply any update to the min time duration box value.
    update_min_t_in_file(min_time_duration_before_box.value,
                         min_time_duration_after_box.value)
    if not selected_param_value:
        return

    cur_selected_param = selected_param_value
    try:
        parameter_obj.set_currently_selected_param(selected_param_value)
    except:
        common.logger.error("Parameter name(%s) not found in list; unexpected")

    param_min_time_duration_before, param_min_time_duration_after = get_param_min_time_duration()
    param_window_duration, param_start_timestamp_series = parameter_obj.get_param_values(cur_selected_param)
    refresh_param_values_ui(param_window_duration, param_start_timestamp_series)


def refresh_param_names_combo_box(parameter_obj):
    param_name_list = parameter_obj.get_param_name_list()
    param_names_combo_box.clear()
    for param_name in param_name_list:
        param_names_combo_box.append(param_name)

    param_names_combo_box.show()

def refresh_param_values_ui(param_window_duration, param_start_timestamp_series):
    set_time_window_duration_box_value(param_window_duration)
    set_min_time_duration_box_value()
    refresh_ts_list_box(param_start_timestamp_series)


def refresh_ts_list_box(ts_series):
    ts_series_list_box.clear()
    for val in ts_series:
        ts_series_list_box.append(val)


def set_min_time_duration_box_value():
    global param_min_time_duration_before, param_min_time_duration_after

    min_time_duration_before_box.value = param_min_time_duration_before
    min_time_duration_after_box.value = param_min_time_duration_after


def set_time_window_duration_box_value(param_window_duration):
    time_window_duration_box.value = param_window_duration


def only_process_sel_param():
    global only_process_cur_param

    if only_process_cur_param_box.value == 1:
        only_process_cur_param = True
    else:
        only_process_cur_param = False


def output_folder_with_file_name():
    global create_output_folder_with_file_name

    if output_folder_with_file_name_box.value == 1:
        create_output_folder_with_file_name = True
    else:
        create_output_folder_with_file_name = False


def csv_file_name_to_output():
    global add_csv_file_name_to_output

    if csv_file_name_to_output_box.value == 1:
        add_csv_file_name_to_output = True
    else:
        add_csv_file_name_to_output = False


if __name__ == "__main__":
    """
    Main entry point
    """

    progress = loghandler()
    logging.basicConfig(filename=OUTPUT_LOG_FILE,
                        level=logging.DEBUG, format='')
    common.logger = logging.getLogger(__name__)
    common.logger.addHandler(progress)
    argv = sys.argv[1:]
    console_mode = False
    separate_files = False
    output_folder = None

    # TODO: Fix this help.
    try:
        opts, args = getopt.getopt(argv, "vhco:d:i:s")
    except getopt.GetoptError as e:
        common.logger.error("USAGE ERROR: %s", e)
        print_help()
    for opt, arg in opts:
        if opt == "-h":
            print_help()
        elif opt in ("-i"):
            input_dir = arg
        elif opt in ("-o"):
            output_folder = arg
        elif opt in ("-v"):
            logging.basicConfig(level=logging.DEBUG)
        elif opt in ("-s"):
            separate_files = True
        elif opt in ("-c"):
            console_mode = True
        else:
            print_help()

    if console_mode:
        main(input_dir, separate_files, output_folder)
        sys.exit()

    # Main app
    app = App("", height=900, width=900)

    # App name box
    title = Text(app, text="Binary Data Processing App",
                 size=16, font="Arial Bold", width=30)
    title.bg = "white"
    line()

    # Select input folder button
    Text(app, "Select Input Folder --> Check Parameters --> Process",
         font="Verdana bold")

    line()
    input_folder_button = PushButton(
        app, command=select_input_dir, args=[parameter_obj], text="Input Folder", width=26)
    input_folder_button.tk.config(font=("Verdana bold", 14))
    # Box to display the input folder
    line()
    input_dir_text_box = TextBox(app)
    # Non editable
    input_dir_text_box.disable()
    input_dir_text_box.width = INPUT_FOLDER_NAME_BOX_MAX_WIDTH
    input_dir_text_box.font = "Verdana bold"
    input_dir_text_box.text_size = 14
    line()

    # Separate files from input folder
    process_button = PushButton(
        app, text="Separate files from input folder (Anymaze)", command=process_separate_input, width=35)
    process_button.tk.config(font=("Verdana bold", 14))
    line()

    # Box to display the the timesfhit message
    timeshift_msg_box = Text(app)
    timeshift_msg_box.value = "Specify timeshift value in the input files, if applies"
    timeshift_msg_box.text_color = "Red"
    line()

    # Master parameter box
    param_box = Box(app, layout="grid")
    cnt = 0
    param_title_box = TitleBox(param_box, text="", grid=[0, cnt])
    param_title = Text(param_title_box, text="Parameters",
                       size=10, font="Arial Bold", align="left", color="orange")

    # Parameter names combo box (grid high to keep it on the right side)
    param_names_combo_box = Combo(
        param_box, options=[], grid=[5, cnt], command=select_param)
    param_names_combo_box.clear()
    param_names_combo_box.text_color = "orange"
    param_names_combo_box.font = "Arial Bold"
    # param_names_combo_box.hide()
    cnt += 1

    # Boxes related to showing parameters
    min_time_duration_label_box = Text(
        param_box, text=PARAM_UI_MIN_TIME_DURATION_CRITERIA_TEXT, grid=[0, cnt], align="left")
    cnt += 1

    min_time_duration_before_label_box = Text(
        param_box, text=PARAM_UI_MIN_BEFORE_TIME_DURATION_CRITERIA_TEXT, grid=[0, cnt], align="left")
    min_time_duration_before_box = TextBox(
        param_box, text="", grid=[1, cnt], align="left")

    min_time_duration_after_label_box = Text(
        param_box, text=PARAM_UI_MIN_AFTER_TIME_DURATION_CRITERIA_TEXT, grid=[2, cnt], align="left")
    min_time_duration_after_box = TextBox(
        param_box, text="", grid=[3, cnt], align="left")
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
        param_box, pd.Series(dtype=np.float64), scrollbar=True,
        grid=[1, cnt], align="left", width=80, height=125)
    cnt += 1

    # Open & update parameters file button
    center_box = Box(app, layout="grid")
    open_params_button = PushButton(center_box, command=open_params_file, args=[parameter_obj],
                                    text="Open parameters file", grid=[0, 1], width=17, align="left")
    open_params_button.tk.config(font=("Verdana bold", 10))
    update_params_button = PushButton(center_box, text="Refresh", command=parse_cur_param_file,
                                      args=[parameter_obj], grid=[1, 1], width=17, align="left")
    update_params_button.tk.config(font=("Verdana bold", 10))
    line()

    # Process input button
    process_button = PushButton(app, text="Process", command=process, width=26)
    process_button.tk.config(font=("Verdana bold", 14))
    only_process_cur_param_box = CheckBox(
        app, text="Only process the selected parameter", command=only_process_sel_param)
    only_process_cur_param_box.value = only_process_cur_param
    output_folder_with_file_name_box = CheckBox(
        app, text="Create additional output folder with file name (FreezeFrame)", command=output_folder_with_file_name)
    output_folder_with_file_name_box.value = create_output_folder_with_file_name
    csv_file_name_to_output_box = CheckBox(
        app, text="Add the csv file name to output file name (Anymaze)", command=csv_file_name_to_output)
    csv_file_name_to_output_box.value = add_csv_file_name_to_output
    line()

    # Browse output folder button
    browse_output_folder_button = PushButton(
        app, command=open_output_folder, text="Browse output folder", width=20)
    browse_output_folder_button.tk.config(font=("Verdana bold", 10))
    line()

    # New Result window
    rwin = Window(app, title="Result Window",
                  visible=False, height=700, width=800)

    # Title box
    rwin_title = Text(rwin, text="Results",
                      size=16, font="Arial Bold", width=25)
    rwin_title.bg = "white"
    line_r(rwin)
    result_text_box = Text(rwin, text="")
    line_r(rwin)

    # Grid to hold the various lists
    rlist_box = Box(rwin, layout="grid")
    cnt = 0
    rwin_success_title = Text(
        rlist_box, text="Successful files:", grid=[0, cnt])
    rwin_unsuccess_title = Text(
        rlist_box, text="Unsuccessful files:", grid=[5, cnt])
    rwin_without_shift = Text(
        rlist_box, text="Files without shift:", grid=[10, cnt])
    cnt += 1
    result_success_list_box = ListBox(
        rlist_box, [], scrollbar=True, grid=[0, cnt])
    result_unsuccess_list_box = ListBox(
        rlist_box, [], scrollbar=True, grid=[5, cnt])
    result_withoutshift_list_box = ListBox(
        rlist_box, [], scrollbar=True, grid=[10, cnt])
    cnt += 1
    line_r(rwin)

    # Details log message box
    r_log_title_box = Text(rwin, text="Detailed log messages:")
    r_log_box = TextBox(rwin, text="", height=200, width=500,
                        multiline=True, scrollbar=True)
    # Non editable
    r_log_box.disable()

    # Results window will be showed once files have been processed
    rwin.hide()

    # Display the app
    app.display()

    logging.shutdown()


"""
------------------------------------------------------------
                Unit Tests
------------------------------------------------------------
"""
input_data1 = {
    common.INPUT_COL0_TS:     [1, 2, 3, 4, 5, 10, 20,  30,  100, 200, 300, 310],
    common.INPUT_COL1_MI:     [1, 2, 3, 4, 5, 6,  7.1, 8.2, 9.3, 10,  11,  20],
    common.INPUT_COL2_FREEZE: [0, 0, 1, 0, 0, 1,  0,   0,   0,   0,   1,   0]
}

output_data1 = {
    OUTPUT_COL0_TS:        [1.0,   3.0,   4.0,   10.0,  20.0,   300.0],
    OUTPUT_COL1_MI:        [1,     3,     4,     6,     7.1,    11],
    OUTPUT_COL2_MI_AVG:    [1.5,   1.5,   4.5,   4.5,   8.65,   8.65],
    OUTPUT_COL3_FREEZE_TP: [ONE_TO_ZERO, ZERO_TO_ONE,
                            ONE_TO_ZERO, ZERO_TO_ONE, ONE_TO_ZERO, ZERO_TO_ONE]
}

output_data1_min_t_1_before = {
    OUTPUT_COL0_TS:        [1.0,   3.0,   4.0,   10.0,  20.0],
    OUTPUT_COL1_MI:        [1,     3,     4,     6,     7.1],
    OUTPUT_COL2_MI_AVG:    [1.5,   1.5,   4.5,   4.5,   8.65],
    OUTPUT_COL3_FREEZE_TP: [ONE_TO_ZERO, ZERO_TO_ONE,
                            ONE_TO_ZERO, ZERO_TO_ONE, ONE_TO_ZERO]
}

output_data1_min_t_4_after = {
    OUTPUT_COL0_TS:        [4.0,   10.0,  20.0],
    OUTPUT_COL1_MI:        [4,     6,     7.1],
    OUTPUT_COL2_MI_AVG:    [4.5,   4.5,   8.65],
    OUTPUT_COL3_FREEZE_TP: [ONE_TO_ZERO, ZERO_TO_ONE, ONE_TO_ZERO]
}

output_data1_min_t_5_before = {
    OUTPUT_COL0_TS:        [10.0,  20.0],
    OUTPUT_COL1_MI:        [6,     7.1],
    OUTPUT_COL2_MI_AVG:    [4.5,   8.65],
    OUTPUT_COL3_FREEZE_TP: [ZERO_TO_ONE, ONE_TO_ZERO]
}

test_p1 = {
    Parameters.PARAM_TIME_WINDOW_START_LIST: [1,  100.0,   300.0],
    Parameters.PARAM_TIME_WINDOW_DURATION:   [10, np.nan,  np.nan]
}

out_p1 = {
    OUTPUT_COL0_TS:        [1.0,   3.0,   4.0,   10.0,  300.0],
    OUTPUT_COL1_MI:        [1.0,   3.0,   4.0,   6.0,   11.0],
    OUTPUT_COL2_MI_AVG:    [1.5,   1.5,   4.5,   4.5,   8.65],
    OUTPUT_COL3_FREEZE_TP: [ONE_TO_ZERO, ZERO_TO_ONE,
                            ONE_TO_ZERO, ZERO_TO_ONE, ZERO_TO_ONE]
}

out_p1_nop = {
    OUTPUT_COL0_TS:        [20.0],
    OUTPUT_COL1_MI:        [7.1],
    OUTPUT_COL2_MI_AVG:    [8.65],
    OUTPUT_COL3_FREEZE_TP: [ONE_TO_ZERO]
}


# Run these tests using `python -m unittest .\binary.py`
class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        global param_name_list, param_df_list

        self.input_df1 = pd.DataFrame(input_data1)
        reset_all_parameters()
        self.p1_df = pd.DataFrame(test_p1)

    def validate_df(self, df, expected_data):
        expected_df = pd.DataFrame(expected_data)
        self.assertEqual(expected_df.equals(df), True)

    def validate_min_t(self, min_t_before, min_t_after, df, exp_out):
        out_df = apply_min_time_duration_criteria(
            min_t_before, min_t_after, df)
        # print(out_df)
        out_df.reset_index(drop=True, inplace=True)
        self.validate_df(out_df, exp_out)

    def test_process_input_df(self):
        exp_out_df = pd.DataFrame(output_data1)
        result, out_df = process_input_df(self.input_df1)
        self.assertEqual(result, True)
        self.assertEqual(exp_out_df.equals(out_df), True)
        self.validate_min_t(1, 0, out_df, output_data1_min_t_1_before)
        self.validate_min_t(5, 0, out_df, output_data1_min_t_5_before)
        self.validate_min_t(0, 4, out_df, output_data1_min_t_4_after)

    def test_param_processing(self):
        param_name_list.append("test_p1")
        param_df_list.append(self.p1_df)
        out_df = process_input_df(self.input_df1)[1]
        nop_df = out_df[:]
        temp_out_df, nop_df = process_param(0, out_df, nop_df)
        temp_out_df.reset_index(drop=True, inplace=True)
        self.validate_df(temp_out_df, out_p1)
        self.validate_df(nop_df, out_p1_nop)
