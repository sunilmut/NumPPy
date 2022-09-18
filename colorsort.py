#!/usr/bin/python

from asyncio.log import logger
import sys
import xlrd
import getopt
from collections import defaultdict
import xlsxwriter
import logging as log
from nested_dict import nested_dict
import logging
import sys
import os
from guizero import App, CheckBox, PushButton, Text, TextBox
import matplotlib

# constants
OUTPUT_LOG_FILE = "output.txt"

# globals
output_file = ""
input_folder = ""
logger = None
input_file = ""

# dictionary to store the parsed input data
# this is a 3 level nested dictionary such as:
# - Each color per column per sheet from the input needs
#   to go to a new column in the output.
# - The same color from all the different input sheets should
#   go to the same sheet in the output.
# input_dic_parsed[sheet number][background color][column] = []
input_dic_parsed = nested_dict(3, list)

# A map that track the output sheet number for a given color
color_to_sheet_num_map = {}

# A record of all the colors seen, to track the column index
# for any given color, for the output.
# Requirement: Each unique color in an input column should go
# to the same column in the output.
cur_column_per_color = {}

break_on_white = False


def parse_input_workbook(in_wb, sheet, sheet_num, break_on_white):
    """
    Parse the given sheet from the input workbook and store
    the data in the parsed dictionary.
    in_wb - input workbook
    sheet - the sheet to process
    sheet_num - the sheet number
    break_on_white - break the color into a new column when seeing
    a white space.
    """
    global input_dic_parsed, cur_column_per_color

    from xlrd.sheet import ctype_text
    # Get the total number of rows and columns in this sheet
    rows, cols = sheet.nrows, sheet.ncols
    # print "Number of rows: %s   number of cols: %s" % (rows, cols)
    for col in range(0, cols):  # Iterate through columns in this sheet
        color_seen_in_this_col = {}
        # For each column, iterate through the rows in that column
        last_non_white_bgx = -1
        white_space_encountered = False
        for row in range(0, rows):
            col_type = sheet.cell_type(row, col)
            # Get cell details of [row, col]
            cell_obj = sheet.cell(row, col)
            xfx = sheet.cell_xf_index(row, col)
            xf = in_wb.xf_list[xfx]
            bgx = xf.background.pattern_colour_index
            if col_type is xlrd.XL_CELL_EMPTY or cell_obj == '' or bgx == 64:
                logging.debug('[%d,%d] empty cell' % (row + 1, col + 1))
                white_space_encountered = True
                continue  # skip empty cells
            # if bgx == 64:
            #   logging.debug('[%d,%d] cell_obj: [%s] white cell, skipping...' % (row + 1, col + 1, cell_obj))
            #   continue # skip white
            if break_on_white and white_space_encountered and last_non_white_bgx != -1:
                logging.debug('breaking on white for color: %d' %
                              (last_non_white_bgx))
                cur_column_per_color[last_non_white_bgx] += 1
            last_non_white_bgx = bgx
            white_space_encountered = False
            # record all the colors that are seen in this column
            if bgx not in color_seen_in_this_col.keys():
                color_seen_in_this_col[bgx] = 1
            if bgx not in cur_column_per_color.keys():
                cur_column_per_color[bgx] = -1
            column = cur_column_per_color[bgx]
            column += 1
            color_map = in_wb.colour_map[bgx]
            logging.debug('[%s,%s] cell_obj: [%s] [%s], column: %d' %
                          (row + 1, col + 1, cell_obj, bgx, column))
            input_dic_parsed[sheet_num][bgx][column].append(cell_obj.value)
        # logging.debug('colors in this col %s', color_seen_in_this_col)
        # for all the colors that were there in this column, increment
        # the column count, so that the next hit of this color can
        # go to the next output column.
        for colors in color_seen_in_this_col:
            cur_column_per_color[colors] += 1
        logging.debug("current column per color %s" % (cur_column_per_color))
    logging.debug('parsed dict %s' % (input_dic_parsed))


def generate_output(out_wb, in_wb):
    global cur_column_per_color, input_dic_parsed

    # cur_column_per_color = {}
    cell_format = out_wb.add_format()
    for sheet_num in input_dic_parsed:
        for color in list(input_dic_parsed[sheet_num]):
            logging.debug('color %s, sheet_num: %d', color, sheet_num)
            if color not in cur_column_per_color.keys():
                cur_column_per_color[color] = 0
            # print 'Color code: %s, col is %s' % (color, col)
            for column in input_dic_parsed[sheet_num][color]:
                row = 0
                for value in input_dic_parsed[sheet_num][color][column]:
                    logging.debug('column %s, value: %s' % (column, value))
                    sheet_name = "Sheet" + str(color_to_sheet_num_map[color])
                    sheet = out_wb.get_worksheet_by_name(sheet_name)
                    if (row == 0):
                        # first entry is the input sheet name
                        input_sheet = in_wb.sheet_by_index(sheet_num)
                        sheet.write(row, column, input_sheet.name, cell_format)
                        row += 1
                    sheet.write(row, column, value, cell_format)
                    row += 1


def write_output_file(out_wb):
    while (True):
        try:
            out_wb.close()
            break
        except xlsxwriter.exceptions.FileCreateError:
            s_continue = app.yesno(
                "Permssion Denied", "Permission denied while writing to the output file. Please close the file if it is open in Excel.\nRetry? Click yes to retry or no to abort")

            if s_continue != True:
                return False

    return True


def bg_to_hex(rgb):
    (r, g, b) = rgb
    return ('#{:X}{:X}{:X}').format(r, g, b)


def sort(input_file, output_file, break_on_ws):
    global logger, color_to_sheet_num_map, input_dic_parsed, cur_column_per_color

    input_dic_parsed.clear()
    cur_column_per_color.clear()

    logger.info("Processing input file: %s, break on whitespace: %d",
                input_file, break_on_ws)

    color_to_sheet_num_map.clear()
    in_wb = xlrd.open_workbook(input_file, formatting_info=True)
    nsheets = in_wb.nsheets
    for sheet_num in range(0, nsheets):
        sheet = in_wb.sheet_by_index(sheet_num)
        logging.info('Processing sheet: %s' % sheet.name)
        parse_input_workbook(in_wb, sheet, sheet_num, break_on_ws)

    out_wb = xlsxwriter.Workbook(output_file)
    logging.info('Total distinct sheets = %s', len(input_dic_parsed))
    color_sheet_map_count = 1
    for sheet_num in input_dic_parsed:
        for color in input_dic_parsed[sheet_num]:
            if color not in color_to_sheet_num_map.keys():
                ws = out_wb.add_worksheet()
                (r, g, b) = in_wb.colour_map[color]
                hex = matplotlib.colors.to_hex((r / 255, g / 255, b / 255))
                ws.set_tab_color(hex)
                color_to_sheet_num_map[color] = color_sheet_map_count
                color_sheet_map_count += 1

    generate_output(out_wb, in_wb)
    logging.info('Output file written: %s' % output_file)
    return write_output_file(out_wb)


"""
------------------------------------------------------------
                UI related stuff
------------------------------------------------------------
"""


def line():
    Text(app, "--------------------------------------------------------------------------")


def select_input_file():
    global input_file, input_folder, output_file

    if input_folder:
        open_folder = input_folder
    else:
        open_folder = "."
    input_file = app.select_file(folder=open_folder)
    if not input_file:
        return

    # Remember the input folder for next time.
    input_folder = os.path.dirname(input_file)
    file_name_without_ext, ext = os.path.splitext(
        os.path.basename(input_file))
    input_file_text_box.value = os.path.basename(input_file)
    output_file = os.path.join(
        input_folder, "o_" + file_name_without_ext + ".xlsx")
    open_output_file_button.enabled = False


def break_on_whitespace_selection():
    global break_on_white

    if break_on_ws_checkbox.value == 1:
        break_on_white = True
    else:
        break_on_white = False


def open_output_file():
    global output_file

    if not output_file:
        app.warn(
            "Uh oh!", "Output file does not exist yet. Please select an input file and sort to create an output file first!")
        return

    os.startfile(output_file)


def sort_cick():
    global input_file, output_file, break_on_white

    open_output_file_button.enabled = False
    if not input_file:
        app.warn(
            "Uh oh!", "No input file selected to color sort. Please select an input excel file and try again.")
        return

    if sort(input_file, output_file, break_on_white) == True:
        open_output_file_button.enabled = True
    else:
        open_output_file_button.enabled = False


if __name__ == "__main__":
    """
    Main entry point
    """

    logging.basicConfig(filename=OUTPUT_LOG_FILE,
                        level=logging.INFO, filemode='w+', format='')
    logger = logging.getLogger(__name__)

    # Main app
    app = App("",  height=400, width=400)

    # App name box
    title = Text(app, text="Excel color sort",
                 size=16, font="Arial Bold", width=30)
    title.bg = "white"
    line()

    # Select input folder button
    Text(app, "Select Excel File --> Sort", font="Verdana bold")
    line()
    input_file_button = PushButton(
        app, command=select_input_file, text="Select Excel File", width=26)
    input_file_button.tk.config(font=("Verdana bold", 14))
    input_file_button.bg = "#ff9933"
    input_file_button.text_color = "white"

    # Box to display the input folder
    line()
    input_file_text_box = TextBox(app)
    # Non editable
    input_file_text_box.disable()
    input_file_text_box.width = 26
    input_file_text_box.font = "Verdana bold"
    input_file_text_box.text_size = 12
    line()

    # Process input button
    process_button = PushButton(
        app, text="Color Sort", command=sort_cick, width=26)
    process_button.tk.config(font=("Verdana bold", 14))
    process_button.bg = "#0099cc"
    process_button.text_color = "white"
    break_on_ws_checkbox = CheckBox(
        app, text="Break on whitespace into next column", command=break_on_whitespace_selection)
    break_on_ws_checkbox.value = 1
    line()

    # Browse output folder button
    open_output_file_button = PushButton(
        app, command=open_output_file, text="Open output file", width=20)
    open_output_file_button.tk.config(font=("Verdana bold", 10))
    open_output_file_button.enabled = False
    line()

    # Display the app
    app.display()
