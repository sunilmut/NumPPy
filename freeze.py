#!/usr/bin/python

import sys
import csv
import getopt
import logging
import os
import pandas as pd
from pandas.api.types import is_numeric_dtype
import glob

# Constants:
# Number of initial rows to skip.
num_initial_rows_to_skip = 3

zero_to_one = "0 to 1"
one_to_zero = "1 to 0"

# input coloumns
input_col1 = "Timestamp"
input_col2 = "Motion Index"
input_col3 = "Freeze"

# output columns
output_col1 = "Timestamp"
output_col2 = "Motion Index"
output_col3 = "Avg of MI"
output_col4 = "Freezing TurnPoints"

# output directory name
output_dir_name = "output"


def parse_input_workbook(input_file, output_file):
    in_col_names = [input_col1, input_col2, input_col3]

    out_col_names = [output_col1, output_col2, output_col3, output_col4]

    df = pd.read_csv(input_file, names=in_col_names,
                     skiprows=num_initial_rows_to_skip)
    out_df = pd.DataFrame(columns=out_col_names)

    # Do some basic format checking. All input fields are expected
    # to be numeric in nature.
    if not (
        is_numeric_dtype(df[input_col1])
        and is_numeric_dtype(df[input_col2])
        and is_numeric_dtype(df[input_col3])
    ):
        print("Invalid input file format: " + input_file)
        return

    # Freeze column is supposed to be binary (0 or 1)
    if df[input_col3].min() < 0 or df[input_col3].max() > 1:
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

        # print("processing ", idx + num_initial_rows_to_skip + 1)

        # For output, we only care about rows where there is a transition
        #  of freeze value. i.e. [0->1] or [1->0]
        if row.values[2] != prev_freeze:
            # First thing is to capture the current values.
            # print('', idx + num_initial_rows_to_skip + 1, sum, itr, sum/itr)
            if prev_freeze == 0:
                freeze = zero_to_one
            else:
                freeze = one_to_zero
            df = pd.DataFrame(
                {
                    output_col1: [row.values[0]],
                    output_col2: [row.values[1]],
                    output_col3: [sum / itr],
                    output_col4: [freeze],
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


def main(argv):
    input_folder_or_file = ""
    input_files = []
    output_folder = ""
    output_file = ""

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
        output_folder = os.path.join(output_folder, output_dir_name)
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


if __name__ == "__main__":
    main(sys.argv[1:])
