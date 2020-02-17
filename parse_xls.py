#!/usr/bin/python

import sys
import xlrd
import getopt
from collections import defaultdict
import xlsxwriter
import logging as log

def nested_dict(n, type):
   if n == 1:
      return defaultdict(type)
   else:
      return defaultdict(lambda: nested_dict(n - 1, type))

# dictionary to store the parsed input data
# this is a 2 level nested dictionary such as
# input_dic_parsed[sheet number][background color] = []
input_dic_parsed = nested_dict(100, float)

# distinct color to sheet number map
color_to_sheet_num_map = {}

# parses the input workbook and stores the data in the input dictionary
def parse_input_workbook(in_wb, sheet, sheet_num):
   from xlrd.sheet import ctype_text
   # Get the total number of rows and columns in this sheet
   rows, cols = sheet.nrows, sheet.ncols
   # print "Number of rows: %s   number of cols: %s" % (rows, cols)
   for row in range(0, rows):  # Iterate through rows
      for col in range(0, cols):  # Iterate through columns
         col_type = sheet.cell_type(row, col)
         if col_type is xlrd.XL_CELL_EMPTY:
            continue # If emtpy cell, continue
         if col_type is not xlrd.XL_CELL_NUMBER:
            raise Exception("%s[row: %s, col %s] is not of type int or float" % (sheet.name, row, col))
         cell_obj = sheet.cell(row, col)  # Get cell object by row, col
         xfx = sheet.cell_xf_index(row, col)
         xf = in_wb.xf_list[xfx]
         bgx = xf.background.pattern_colour_index
         # print('[%s,%s] cell_obj: [%s] [%s]' % (row + 1, col + 1, cell_obj, bgx))
         if bgx in input_dic_parsed[sheet_num].keys():  # if key is present in the list, just append the value
            input_dic_parsed[sheet_num][bgx].append(cell_obj.value)
         else:
            input_dic_parsed[sheet_num][bgx] = []  # else create a empty list as value for the key
            input_dic_parsed[sheet_num][bgx].append(cell_obj.value)  # now append the value for that key

def generate_output(out_wb):
   cur_column_per_color = {}
   cell_format = out_wb.add_format()
   for sheet_num in input_dic_parsed:
      for color in input_dic_parsed[sheet_num]:
         row = 0
         if color not in cur_column_per_color.keys():
            cur_column_per_color[color] = 0
         col = cur_column_per_color[color]
         # print 'Color code: %s, col is %s' % (color, col)
         for value in input_dic_parsed[sheet_num][color]:
            # print '\t%s' % (value)
            sheet_name = "Sheet" + str(color_to_sheet_num_map[color])
            sheet = out_wb.get_worksheet_by_name(sheet_name)
            sheet.write(row, col, value)
            row += 1
         cur_column_per_color[color] += 1

def print_help():
   print 'extract_xls.py -i <inputfile> -o <outputfile>'
   print 'only works with xls files for input for now'
   print 'close the output file prior to running'
   sys.exit()

def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print_help()
   for opt, arg in opts:
      if opt == '-h':
         print_help()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
      else:
         print_help()

   in_wb = xlrd.open_workbook(inputfile, formatting_info=True)
   nsheets = in_wb.nsheets
   for sheet_num in range(0, nsheets):
      sheet = in_wb.sheet_by_index(sheet_num)
      # print('Sheet name: %s' % sheet.name)
      parse_input_workbook(in_wb, sheet, sheet_num)

   out_wb = xlsxwriter.Workbook(outputfile)
   # print 'total distinct sheets = %s' % (len(input_dic_parsed))
   color_sheet_map_count = 1
   for sheet_num in input_dic_parsed:
      for color in input_dic_parsed[sheet_num]:
         if color not in color_to_sheet_num_map.keys():
            # print('color is:', color)
            out_wb.add_worksheet()
            color_to_sheet_num_map[color] = color_sheet_map_count
            color_sheet_map_count += 1

   # print("color sheet map is : ", color_to_sheet_num_map)
   generate_output(out_wb)
   print('Output file written: %s' % outputfile)
   out_wb.close()

if __name__ == "__main__":
   main(sys.argv[1:])
