#!/usr/bin/python

import sys
import csv
import getopt
import logging
import os
import pandas as pd
from pandas.api.types import is_numeric_dtype
import glob
import guizero
from guizero import App, PushButton, Text, TextBox
import subprocess

# Constants:
# Number of initial rows to skip.
NUM_INITIAL_ROWS_TO_SKIP = 3

ZERO_TO_ONE = "0 to 1"
ONE_TO_ZERO = "1 to 0"

# input coloumns
INPUT_COL0 = "Timestamp"
INPUT_COL1 = "Motion Index"
INPUT_COL2 = "Freeze"

# output columns
OUTPUT_COL0 = "Timestamp"
OUTPUT_COL1 = "Motion Index"
OUTPUT_COL2 = "Avg of MI"
OUTPUT_COL3 = "Freezing TurnPoints"

# output directory name
OUTPUT_DIR_NAME = "output"

# globals
input_dir = ""
output_dir = ""


def apply_duration_criteria(ts_series):
    it = ts_series.iteritems()
    prev_ts = 0
    for idx, val in it:
        if (val - prev_ts > 0.5):
            yield idx
        prev_ts = val


def parse_input_workbook(input_file, out_file_zero_to_one, out_file_one_to_zero):
    in_col_names = [INPUT_COL0, INPUT_COL1, INPUT_COL2]
    out_col_names = [OUTPUT_COL0, OUTPUT_COL1, OUTPUT_COL2, OUTPUT_COL3]
    df = pd.read_csv(input_file, names=in_col_names,
                     skiprows=NUM_INITIAL_ROWS_TO_SKIP)
    out_df = pd.DataFrame(columns=out_col_names)

    # Do some basic format checking. All input fields are expected
    # to be numeric in nature.
    if not (
        is_numeric_dtype(df[INPUT_COL0])
        and is_numeric_dtype(df[INPUT_COL1])
        and is_numeric_dtype(df[INPUT_COL2])
    ):
        print("Invalid input file format: " + input_file)
        return

    # Freeze column is supposed to be binary (0 or 1)
    if df[INPUT_COL2].min() < 0 or df[INPUT_COL2].max() > 1:
        print(
            "Invalid input file format in "
            + input_file
            + ". Column 3 (freeze) value outside bounds (should be 0 or 1)"
        )
        return

    sum = 0
    itr = 0

    # Iterate over all the rows
    for (idx, row) in df.iterrows():
        # Take the freeze value from the first row as the starting freeze
        if idx == 0:
            prev_freeze = row.values[2]

        # For output, we only care about rows where there is a transition
        # of freeze value. i.e. [0->1] or [1->0]
        if row.values[2] != prev_freeze:
            # First thing is to capture the current values.
            if prev_freeze == 0:
                freeze = ZERO_TO_ONE
            else:
                freeze = ONE_TO_ZERO
            df = pd.DataFrame(
                {
                    OUTPUT_COL0: [row.values[0]],
                    OUTPUT_COL1: [row.values[1]],
                    OUTPUT_COL2: [sum / itr],
                    OUTPUT_COL3: [freeze],
                }
            )
            out_df = pd.concat([out_df, df], ignore_index=True, sort=False)

            # Since this is a transition, update the previous freeze value to
            # take the new value.
            prev_freeze = row.values[2]

            # Reset the sum and the iterator every time freeze transitions from [1->0]
            if row.values[2] == 0:
                sum = 0
                itr = 0

        # We need to average the indexes where the freeze is '0'
        if row.values[2] == 0:
            sum += row.values[1]
            itr += 1

    logging.debug(out_df)

    # Apply any minimum time duration criteria
    out_df = out_df.loc[list(apply_duration_criteria(out_df.iloc[:, 0]))]

    # Split the dataframe based on whether it is a [0->1] or [1->0] transitions
    out_zero_to_one_df = pd.DataFrame(columns=out_col_names)
    out_one_to_zero_df = pd.DataFrame(columns=out_col_names)
    out_zero_to_one_df = out_df.loc[out_df[OUTPUT_COL3] == ZERO_TO_ONE]
    out_one_to_zero_df = out_df.loc[out_df[OUTPUT_COL3] == ONE_TO_ZERO]
    out_zero_to_one_df.to_csv(out_file_zero_to_one, index=False)
    out_one_to_zero_df.to_csv(out_file_one_to_zero, index=False)


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


input_folder_or_file = ""


def main(argv, input_folder_or_file):
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
        search_path = input_folder_or_file + "\*.csv"
        for file in glob.glob(search_path):
            input_files.append(file)
    elif os.path.isfile(input_folder_or_file):
        input_files.append(input_folder_or_file)
    else:
        print("The input path is not a valid directory or file: ",
              input_folder_or_file)
        print_help()

    # If an output folder is specified, use it.
    # Else, output folder is '<parent of input file or folder>\output', create it
    if not output_folder:
        output_folder = os.path.dirname(input_folder_or_file)
        output_folder = os.path.join(output_folder, OUTPUT_DIR_NAME)
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

    out_file_zero_to_one = ""
    out_file_one_to_zero = ""
    for input_file in input_files:
        # construct an output file using:
        #     'output_folder\<input file name>_output.csv'
        file_name = os.path.basename(input_file)
        out_file_zero_to_one = os.path.join(
            output_folder, os.path.splitext(
                file_name)[0] + "_output_0_to_1.csv"
        )
        out_file_one_to_zero = os.path.join(
            output_folder, os.path.splitext(
                file_name)[0] + "_output_1_to_0.csv"
        )

        print("\nInput file: ", input_file)
        print("Output folder: ", output_folder)
        parse_input_workbook(
            input_file, out_file_zero_to_one, out_file_one_to_zero)

    return output_folder


def line():
    Text(app, "------------------------------------------------------------")


def get_folder():
    global input_dir
    input_dir.value = app.select_folder()
    output_folder_textbox.value = ""


def open_output_folder():
    global output_dir
    if not output_dir:
        app.warn(
            "Uh oh!", "Output folder can be browsed once input is processed. Select input folder and process first!")
        return
    subprocess.Popen(f'explorer /select,{output_dir}')


def process():
    global output_dir
    global input_dir
    if not input_dir.value:
        app.warn(
            "Uh oh!", "No input folder specified. Please select an input folder and run again!")
        return
    output_dir = main(sys.argv[1:], input_dir.value)
    output_folder_textbox.value = ("Output folder: " + output_dir)


if __name__ == "__main__":
    app = App("Freeze Data Processing App")
    line()
    instruction = Text(
        app, "Select Input Folder and then process")
    line()
    PushButton(app, command=get_folder, text="Input Folder")
    input_dir = Text(app)
    line()
    button = PushButton(app, text="Process", command=process)
    line()
    output_folder_textbox = Text(app, text="", color="red")
    PushButton(app, command=open_output_folder, text="Browse output folder")
    line()
    app.display()
