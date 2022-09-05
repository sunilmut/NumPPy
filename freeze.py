#!/usr/bin/python

import sys
import csv
import getopt
import logging
import os
import pandas as pd
from pandas.api.types import is_numeric_dtype

# Constants:
# Number of initial rows to skip.
num_initial_rows_to_skip = 3

zero_to_one = '0 to 1'
one_to_zero = '1 to 0'

# input coloumns
input_col1 = 'Timestamp'
input_col2 = 'Motion Index'
input_col3 = 'Freeze'

# output columns
output_col1 = 'Timestamp'
output_col2 = 'Motion Index'
output_col3 = 'Avg of MI'
output_col4 = 'Freezing TurnPoints'

def parse_input_workbook(input_file, output_file):
   in_col_names = [input_col1, input_col2, input_col3]

   out_col_names = [output_col1, output_col2, output_col3, output_col4]

   df = pd.read_csv(input_file, names=in_col_names, skiprows=num_initial_rows_to_skip)
   out_df = pd.DataFrame(columns=out_col_names)

   # Do some basic format checking. All input fields are expected
   # to be numeric in nature.
   if not (is_numeric_dtype(df[input_col1]) and
           is_numeric_dtype(df[input_col2]) and
           is_numeric_dtype(df[input_col3])):
      print('Invalid input input file format: ' + input_file)
      return

   # Freeze column is supposed to be binary (0 or 1)
   if df[input_col3].min() < 0 or df[input_col3].max() > 1:
      print('Invalid input file format in ' + input_file + '. Column 3 (freeze) value outside bounds (should be 0 or 1)')
      return

   sum = 0
   itr = 0

   # Iterate over all the rows
   for (idx, row) in df.iterrows():
      # Take the freeze value from the first row as the starting freeze
      if idx == 0:
         prev_freeze = row.values[2]

      #print("processing ", idx + num_initial_rows_to_skip + 1)

      # For output, we only care about rows where there is a transition
      #  of freeze value. i.e. [0->1] or [1->0]
      if row.values[2] != prev_freeze:
         # First thing is to capture the current values.
         #print('', idx + num_initial_rows_to_skip + 1, sum, itr, sum/itr)
         if prev_freeze == 0:
            freeze = zero_to_one
         else:
            freeze = one_to_zero
         df = pd.DataFrame({output_col1: [row.values[0]],
                            output_col2: [row.values[1]],
                            output_col3 : [sum/itr],
                            output_col4 : [freeze]})
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
   print('python freeze.py -i <input_file.csv> -o <output_file.csv')
   print('Ex: python freeze.py -i input.csv -o output.csv')
   print('Notes:')
   print('- Close the output file prior to running.')
   sys.exit()

def main(argv):
   input_file = ''
   output_file = ''
   try:
      opts, args = getopt.getopt(argv,"vhi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print_help()
   for opt, arg in opts:
      if opt == '-h':
         print_help()
      elif opt in ("-i", "--ifile"):
         input_file = arg
      elif opt in ("-o", "--ofile"):
         output_file = arg
      elif opt in ("-v"):
         logging.basicConfig(level=logging.DEBUG)
      else:
         print_help()

   # get the input file if one is not provided.
   if input_file == '':
      input_file = input("Enter the name of the input [.csv] file: ")

   # get the output file if one is not provided.
   if output_file == '':
      output_file = input("Enter the name of the output [.csv] file: ")

   # strip the quotes at the start and end, else
   # paths with white spaces won't work.
   input_file = input_file.strip('\"')
   parse_input_workbook(input_file, output_file)

if __name__ == "__main__":
   main(sys.argv[1:])
