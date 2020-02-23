#!/usr/bin/python

import sys
import xlrd
import getopt
from collections import defaultdict
import xlsxwriter
import logging as log
from nested_dict import nested_dict

# dictionary to store the parsed input data
# this is a 3 level nested dictionary such as
# input_dic_parsed[sheet number][background color] = []
input_dic_parsed = nested_dict(3, list)

# distinct color to sheet number map
color_to_sheet_num_map = {}
cur_column_per_color = {}

# parses the input workbook and stores the data in the input dictionary
def parse_input_workbook(in_wb, sheet, sheet_num):
   from xlrd.sheet import ctype_text
   # Get the total number of rows and columns in this sheet
   rows, cols = sheet.nrows, sheet.ncols
   # print "Number of rows: %s   number of cols: %s" % (rows, cols)
   for col in range(0, cols):  # Iterate through columns
      color_seen_in_this_row = {}
      for row in range(0, rows):  # Iterate through rows 
         col_type = sheet.cell_type(row, col)
         if col_type is xlrd.XL_CELL_EMPTY:
            continue # skip empty cells
         cell_obj = sheet.cell(row, col)  # Get cell object by row, col
         xfx = sheet.cell_xf_index(row, col)
         xf = in_wb.xf_list[xfx]
         bgx = xf.background.pattern_colour_index
         if bgx not in color_seen_in_this_row.keys():
            color_seen_in_this_row[bgx] = 1
         if bgx not in cur_column_per_color.keys():
            cur_column_per_color[bgx] = -1
         column = cur_column_per_color[bgx]
         column += 1
         color_map = in_wb.colour_map[bgx]
         if bgx == 64:
            continue # skip white
         if bgx not in color_seen_in_this_row.keys():
            color_seen_in_this_row[bgx]
         print('[%s,%s] cell_obj: [%s] [%s], column: %d' % (row + 1, col + 1, cell_obj, bgx, column))
         input_dic_parsed[sheet_num][bgx][column].append(cell_obj.value)
      print('colors in this col %s', color_seen_in_this_row)
      for colors in color_seen_in_this_row:
         cur_column_per_color[colors] += 1
      print("current column per color %s" % (cur_column_per_color))
   print('parsed dict %s' % (input_dic_parsed))

def generate_output(out_wb):
   cur_column_per_color = {}
   cell_format = out_wb.add_format()
   for sheet_num in input_dic_parsed:
      for color in list(input_dic_parsed[sheet_num]):
         print ('color %s', color)
         if color not in cur_column_per_color.keys():
            cur_column_per_color[color] = 0
         col = cur_column_per_color[color]
         # print 'Color code: %s, col is %s' % (color, col)
         for column in input_dic_parsed[sheet_num][color]:
            row = 0
            for value in input_dic_parsed[sheet_num][color][column]:
               print ('column %s, value: %s' % (column, value))
               sheet_name = "Sheet" + str(color_to_sheet_num_map[color])
               sheet = out_wb.get_worksheet_by_name(sheet_name)
               #.set_bg_color('green')
               sheet.write(row, column, value, cell_format)
               row += 1
         cur_column_per_color[color] += col

def print_help():
   print ('extract_xls.py -i <inputfile.xls> -o <outputfile.xlsx>')
   print ('Only works with xls files for input for now')
   print ('Close the output file prior to running')
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
