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
INPUT_COL1 = "Timestamp"
INPUT_COL2 = "Motion Index"
INPUT_COL3 = "Freeze"

# output columns
OUTPUT_COL1 = "Timestamp"
OUTPUT_COL2 = "Motion Index"
OUTPUT_COL3 = "Avg of MI"
OUTPUT_COL4 = "Freezing TurnPoints"

# output directory name
OUTPUT_DIR_NAME = "output"

# globals
input_dir = ""
output_dir = ""


def parse_input_workbook(input_file, output_file):
    in_col_names = [INPUT_COL1, INPUT_COL2, INPUT_COL3]
    out_col_names = [OUTPUT_COL1, OUTPUT_COL2, OUTPUT_COL3, OUTPUT_COL4]
    df = pd.read_csv(input_file, names=in_col_names,
                     skiprows=NUM_INITIAL_ROWS_TO_SKIP)
    out_df = pd.DataFrame(columns=out_col_names)

    # Do some basic format checking. All input fields are expected
    # to be numeric in nature.
    if not (
        is_numeric_dtype(df[INPUT_COL1])
        and is_numeric_dtype(df[INPUT_COL2])
        and is_numeric_dtype(df[INPUT_COL3])
    ):
        print("Invalid input file format: " + input_file)
        return

    # Freeze column is supposed to be binary (0 or 1)
    if df[INPUT_COL3].min() < 0 or df[INPUT_COL3].max() > 1:
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
        #  of freeze value. i.e. [0->1] or [1->0]
        if row.values[2] != prev_freeze:
            # First thing is to capture the current values.
            if prev_freeze == 0:
                freeze = ZERO_TO_ONE
            else:
                freeze = ONE_TO_ZERO
            df = pd.DataFrame(
                {
                    OUTPUT_COL1: [row.values[0]],
                    OUTPUT_COL2: [row.values[1]],
                    OUTPUT_COL3: [sum / itr],
                    OUTPUT_COL4: [freeze],
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

    prev_ts = 0
    out_zero_to_one_df = pd.DataFrame(columns=out_col_names)
    out_one_to_zero_df = pd.DataFrame(columns=out_col_names)
    # Apply any minimum time duration criteria
    for (idx, row) in out_df.iterrows():
        if (row.values[0] - prev_ts > 0.5):
            if (row.values[3] == ZERO_TO_ONE):
                out_zero_to_one_df = pd.concat(
                    [out_zero_to_one_df, out_df.loc[[idx]]], ignore_index=True, sort=False)
            else:
                out_one_to_zero_df = pd.concat(
                    [out_one_to_zero_df, out_df.loc[[idx]]], ignore_index=True, sort=False)
        prev_ts = row.values[0]

    print("zero to one: ", out_zero_to_one_df)
    print("one to zero: ", out_one_to_zero_df)
    out_df.to_csv(output_file, index=False)


def print_help():
    print("\nHelp/Usage:\n")
    print(
        "python freeze.py -i <input folder or .csv file> -o <output_file.csv> -d <output_directory> -v -h\n"
    )
    print("where:")
    print("-i (required): input folder or .csv file.")
    print("-o (optional): output .csv file.")
    print("-d (optional): output folder.")
    print("-v (optional): run in verbose mode")
    print("-h (optional): print this help")
    print("\nExamples:\n")
    print("\nProcess input file and emit to an output file:")
    print("\tpython freeze.py -i input.csv -o output.csv")
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
    output_file = ""
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
        elif opt in ("-o"):
            output_file = arg
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

    # If an output file or folder is specified, use it.
    # Else, output folder is '<parent of input file or folder>\output', create it
    if not output_file and not output_folder:
        output_folder = os.path.dirname(input_folder_or_file)
        output_folder = os.path.join(output_folder, OUTPUT_DIR_NAME)
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

    out_file = ""
    for input_file in input_files:
        # If an output file is specified, use it.
        # Else, construct an output file using:
        #     'output_folder\<input file name>_output.csv'
        if output_file:
            if os.path.isdir(output_file):
                print("The specified output file is a folder: ", output_file)
                print_help()

            out_file = output_file
        else:
            file_name = os.path.basename(input_file)
            out_file = os.path.join(
                output_folder, os.path.splitext(file_name)[0] + "_output.csv"
            )

        print("\nInput file: ", input_file)
        print("Output file: ", out_file)
        parse_input_workbook(input_file, out_file)

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
