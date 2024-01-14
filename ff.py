import h5py
import sys
import getopt
import logging
import numpy as np
import common
from parameter import *
import os
import pandas as pd
from pandas.api.types import is_integer_dtype
from guizero import App, Box, Combo, ListBox, PushButton, Text, TextBox, TitleBox
import scipy
import glob

# UI related constants
INPUT_FOLDER_NAME_BOX_MAX_WIDTH = 26

# Parameter UI strings
PARAM_UI_TIME_WINDOW_START_TIMES = "Time window start (secs):"
PARAM_UI_MIN_TIME_DURATION_CRITERIA_TEXT = "Min time duration criteria (secs):"
PARAM_UI_MIN_BEFORE_TIME_DURATION_CRITERIA_TEXT = "\t\t    before: "
PARAM_UI_MIN_AFTER_TIME_DURATION_CRITERIA_TEXT = "after: "
PARAM_UI_TIME_WINDOW_DURATION_TEXT = "Time window duration (secs): "

# Other constants
OUTPUT_LOG_FILE = "output.txt"
OUTPUT_COL0_TS = 'Start time (sec)'
OUTPUT_COL1_LEN = 'Bout length (sec)'
OUTPUT_COL2_MI_AVG = 'Motion Index (avg)'
OUTPUT_COL3_DATA_AUC = 'AUC (data)'
OUTPUT_COL4_DATA_AVG = 'z-score'
OUTPUT_COLUMN_NAMES=[OUTPUT_COL0_TS, OUTPUT_COL1_LEN,
                     OUTPUT_COL2_MI_AVG, OUTPUT_COL3_DATA_AUC,
                     OUTPUT_COL4_DATA_AVG]

# output file names
OUTPUT_ZEROS = "0_"
OUTPUT_ONES = "1_"
OUTPUT_NOT = "_Not"
OUTPUT_AUC = "AUC (sum)"
OUTPUT_AUC_SEM = "AUC_SEM"
OUTPUT_Z_SCORE = "z-score (avg)"
OUTPUT_Z_SCORE_SEM = "z-score_SEM"
OUTPUT_SUMMARY_COLUMN_NAMES = [OUTPUT_AUC, OUTPUT_AUC_SEM,
                               OUTPUT_Z_SCORE, OUTPUT_Z_SCORE_SEM]

# Globals
parameter_obj = Parameters()

# Read the values of the given 'key' from the HDF5 file
# into an numpy array. If an 'event' is provided, it will
# be appended to the filepath.
def read_hdf5(event, filepath, key):
    if event:
        event = event.replace("\\","_")
        event = event.replace("/","_")
        op = os.path.join(filepath, event + '.hdf5')
    else:
        op = filepath

    if os.path.exists(op):
        with h5py.File(op, 'r') as f:
            arr = np.asarray(f[key])
    else:
        common.logger.error(f"{op}.hdf5 file does not exist")
        raise Exception('{}.hdf5 file does not exist'.op)

    return arr

def compute_val(cnt, sum, df):
    avg = sum/cnt
    # std error of mean (sem)
    sem_sum = scipy.stats.sem(df.loc[:, OUTPUT_COL3_DATA_AUC])
    sem_avg = scipy.stats.sem(df.loc[:, OUTPUT_COL4_DATA_AVG])

    return avg, sem_sum, sem_avg

def main(input_dir, parameter_obj):
    path = glob.glob(os.path.join(input_dir, 'z_score_*'))
    output_dir = common.get_output_dir(input_dir, '')
    for i in range(len(path)):
        basename = (os.path.basename(path[i])).split('.')[0]
        name_1 = basename.split('_')[-1]
        # TODO: Should NaN be handled?
        z_score = read_hdf5('', path[i], 'data')
        ts = read_hdf5('timeCorrection_' + name_1, input_dir, 'timestampNew')
        csv_path = glob.glob(os.path.join(input_dir, '*.csv'))
        for csv_file in csv_path:
            timeshift_val, num_rows_processed = common.get_timeshift_from_input_file(csv_file)
            success, binary_df = common.parse_input_file_into_df(csv_file,
                                                    common.NUM_INITIAL_ROWS_TO_SKIP + num_rows_processed)
            if not success:
                common.logger.warning("Skipping CSV file (%s) as it is well formed")
                continue

            #print(binary_df)
            csv_basename = (os.path.basename(csv_file)).split('.')[0]
            # Create output folder specific for this csv file.
            this_output_folder = os.path.join(output_dir, csv_basename)
            #common.logger.debug("Output folder: %s", this_output_folder)
            if not os.path.isdir(this_output_folder):
                os.mkdir(this_output_folder)

            # We want to generate without any parameters as well. So start with
            # no parameters and append the parameter list.
            param_name_list = [""]
            #param_name_list = []
            param_name_list.extend(parameter_obj.get_param_name_list())
            for param in param_name_list:
                # Process the data and write out the results
                print("processing param: ", param)
                success, results = process(parameter_obj,
                                           param,
                                           binary_df,
                                           timeshift_val,
                                           z_score,
                                           ts)
                if not success:
                    continue

                auc_0s_sum = results[0]
                auc_0s_cnt = results[1]
                out_df_0s = results[2]
                auc_1s_sum = results[3]
                auc_1s_cnt = results[4]
                out_df_1s = results[5]
                auc_0s_sum_out = results[6]
                auc_0s_cnt_out = results[7]
                out_df_0s_out = results[8]
                auc_1s_sum_out = results[9]
                auc_1s_cnt_out = results[10]
                out_df_1s_out = results[11]

                param_ext = ""
                if not param == "":
                    param_ext = "_" + param

                # 0's, in the param
                if auc_0s_cnt > 0:
                    auc_0s_avg, sem_auc_0s_sum, sem_auc_0s_avg = compute_val(auc_0s_sum, auc_0s_cnt, out_df_0s)
                    df_0s_summary = pd.DataFrame(columns=OUTPUT_SUMMARY_COLUMN_NAMES)
                    df_0s_summary.loc[len(df_0s_summary.index)] = [auc_0s_sum,
                                                                sem_auc_0s_sum,
                                                                auc_0s_avg,
                                                                sem_auc_0s_avg]
                    out_0_file = os.path.join(this_output_folder,
                                            OUTPUT_ZEROS + csv_basename + param_ext + common.CSV_EXT)
                    df_0s_summary.to_csv(out_0_file, mode='w', index=False, header=True)
                    out_df_0s.to_csv(out_0_file, mode='a', index=False, header=True)
                    print("0 df: ", out_df_0s)
                    print("0s sum: ", auc_0s_sum, " avg: ", auc_0s_avg, " SEM_AUC: ", sem_auc_0s_sum, " SEM_AVG: ", sem_auc_0s_avg)

                # 0's, outside the param
                if auc_0s_cnt_out > 0:
                    auc_0s_avg_out, sem_auc_0s_sum_out, sem_auc_0s_avg_out = compute_val(auc_0s_sum_out,
                                                                                         auc_0s_cnt_out,
                                                                                         out_df_0s_out)
                    df_0s_out_summary = pd.DataFrame(columns=OUTPUT_SUMMARY_COLUMN_NAMES)
                    df_0s_out_summary.loc[len(df_0s_out_summary.index)] = [auc_0s_sum_out,
                                                                           sem_auc_0s_sum_out,
                                                                           auc_0s_avg_out,
                                                                           sem_auc_0s_avg_out]
                    out_0_file = os.path.join(this_output_folder,
                                            OUTPUT_ZEROS + csv_basename + OUTPUT_NOT + param_ext + common.CSV_EXT)
                    df_0s_out_summary.to_csv(out_0_file, mode='w', index=False, header=True)
                    out_df_0s_out.to_csv(out_0_file, mode='a', index=False, header=True)
                    print("0 df [out]: ", out_df_0s_out)
                    print("0s [out] sum: ", auc_0s_sum_out, " avg: ", auc_0s_avg_out, " SEM_AUC: ", sem_auc_0s_sum_out, " SEM_AVG: ", sem_auc_0s_avg_out)

                # 1's, in the param
                if auc_1s_cnt > 0:
                    auc_1s_avg, sem_auc_1s_sum, sem_auc_1s_avg = compute_val(auc_1s_sum, auc_1s_cnt, out_df_1s)
                    df_1s_summary = pd.DataFrame(columns=OUTPUT_SUMMARY_COLUMN_NAMES)
                    df_1s_summary.loc[len(df_1s_summary.index)] = [auc_1s_sum,
                                                                sem_auc_1s_sum,
                                                                auc_1s_avg,
                                                                sem_auc_1s_avg]
                    out_1_file = os.path.join(this_output_folder,
                                                OUTPUT_ONES + csv_basename + param_ext + common.CSV_EXT)
                    df_1s_summary.to_csv(out_1_file, mode='w', index=False, header=True)
                    out_df_1s.to_csv(out_1_file, mode='a', index=False, header=True)
                    print("1 df: ", out_df_1s)
                    print("1s sum: ", auc_1s_sum, " avg: ", auc_1s_avg, " SEM_AUC: ", sem_auc_1s_sum, " SEM_AVG: ", sem_auc_1s_avg)

                # 1's, outside the param
                if auc_1s_cnt_out > 0:
                    auc_1s_avg_out, sem_auc_1s_sum_out, sem_auc_1s_avg_out = compute_val(auc_1s_sum_out,
                                                                                         auc_1s_cnt_out,
                                                                                         out_df_1s_out)
                    df_1s_out_summary = pd.DataFrame(columns=OUTPUT_SUMMARY_COLUMN_NAMES)
                    df_1s_out_summary.loc[len(df_1s_out_summary.index)] = [auc_1s_sum_out,
                                                                           sem_auc_1s_sum_out,
                                                                           auc_1s_avg_out,
                                                                           sem_auc_1s_avg_out]
                    out_0_file = os.path.join(this_output_folder,
                                            OUTPUT_ONES + csv_basename + OUTPUT_NOT + param_ext + common.CSV_EXT)
                    df_1s_out_summary.to_csv(out_0_file, mode='w', index=False, header=True)
                    out_df_1s_out.to_csv(out_0_file, mode='a', index=False, header=True)
                    print("1 df [out]: ", out_df_1s_out)
                    print("1s [out] sum: ", auc_1s_sum_out, " avg: ", auc_1s_avg_out, " SEM_AUC: ", sem_auc_1s_sum_out, " SEM_AVG: ", sem_auc_1s_avg_out)

def process(parameter_obj,
            param_name,
            binary_df,
            timeshift_val,
            data,
            ts):

    # Perform some basic checks on the data sets.
    if len(ts) != len(data):
        common.logger.error("Timestamp series length(%d) does not match data series length(%d)",
                            len(ts),
                            len(data))
        return False

    if not binary_df[common.INPUT_COL0_TS].is_monotonic:
        common.logger.error("Binary timestamp values are not sorted.")
        return False, []

    # Make sure the 'binary' column is actually binary.
    if not (
        is_integer_dtype(binary_df[common.INPUT_COL2_FREEZE])
        and binary_df[common.INPUT_COL2_FREEZE].min() == 0
        and binary_df[common.INPUT_COL2_FREEZE].max() == 1
    ):
        common.logger.error("Binary column contains non-binary data.")
        return False, []

    # Timestamp series should be sorted.
    if not np.all(np.diff(ts) >= 0):
        common.logger.error("Timestamp series is not sorted.")
        return False, []

    index_start = -1
    row_count = binary_df.shape[0]
    auc_0s_sum = 0
    auc_0s_cnt = 0
    mi_0s_sum = 0
    mi_0s_cnt = 0
    out_df_0s = pd.DataFrame(columns=OUTPUT_COLUMN_NAMES)
    auc_0s_sum_out = 0
    auc_0s_cnt_out = 0
    mi_0s_sum_out = 0
    mi_0s_cnt_out = 0
    out_df_0s_out = pd.DataFrame(columns=OUTPUT_COLUMN_NAMES)
    auc_1s_sum = 0
    auc_1s_cnt = 0
    mi_1s_sum = 0
    mi_1s_cnt = 0
    out_df_1s = pd.DataFrame(columns=OUTPUT_COLUMN_NAMES)
    auc_1s_sum_out = 0
    auc_1s_cnt_out = 0
    mi_1s_sum_out = 0
    mi_1s_cnt_out = 0
    out_df_1s_out = pd.DataFrame(columns=OUTPUT_COLUMN_NAMES)
    print("timeshift_val: ", timeshift_val)
    #print("binary file has rows", row_count)
    for index, row in binary_df.iterrows():
        if index_start == -1:
            index_start = index
            cur_binary_value = binary_df.iloc[index_start][common.INPUT_COL2_FREEZE]

        if cur_binary_value == row[common.INPUT_COL2_FREEZE]:
            index_end = index

        # If there is a transition of the freeze value, compute the variables.
        # Note: The last row is also considered as the end of transition.
        if (
            index < row_count - 1 and
            (cur_binary_value == binary_df.iloc[index + 1][common.INPUT_COL2_FREEZE])
        ):
            continue

        ts_start_without_shift = binary_df.iloc[index_start][common.INPUT_COL0_TS]
        ts_start = round(ts_start_without_shift + timeshift_val, Parameters.TIMESTAMP_ROUND_VALUE)
        ts_end_without_shift = binary_df.iloc[index_end][common.INPUT_COL0_TS]
        ts_end = round(ts_end_without_shift + timeshift_val, Parameters.TIMESTAMP_ROUND_VALUE)
        ts_split = parameter_obj.get_ts_series_for_timestamps(param_name, ts_start, ts_end)
        for element_idx, element in enumerate(ts_split):
            ts_start = element[0]
            ts_end = element[1]
            is_inside = element[2]

            binary_df_ts = binary_df[common.INPUT_COL0_TS]
            ts_start_without_shift = round(ts_start - timeshift_val, Parameters.TIMESTAMP_ROUND_VALUE)
            index_start = np.argmax(binary_df_ts >= ts_start_without_shift)
            ts_end_without_shift = round(ts_end - timeshift_val, Parameters.TIMESTAMP_ROUND_VALUE)
            index_end = np.argmax(binary_df_ts > ts_end_without_shift)
            if index_start == index_end:
               index_end += 1
            if index_end == 0:
                index_end = len(binary_df_ts)

            ts_start = round(binary_df.iloc[index_start][common.INPUT_COL0_TS] + timeshift_val, Parameters.TIMESTAMP_ROUND_VALUE)
            ts_end = round(binary_df.iloc[index_end - 1][common.INPUT_COL0_TS] + timeshift_val, Parameters.TIMESTAMP_ROUND_VALUE)

            ts_index_start_for_val = np.argmax(ts >= ts_start)
            if ts_index_start_for_val == 0 and ts_start > ts[len(ts) - 1]:
                common.logger.debug("ts start is out of bounds. ts_start: %f, ts[last]: %f", ts_start, ts[len(ts) - 1])
                break
            ts_index_end_for_val = np.argmax(ts > ts_end)
            # If there is only one element, then include it.
            if ts_index_start_for_val == ts_index_end_for_val:
                ts_index_end_for_val += 1
            if ts_index_end_for_val == 0:
                ts_index_end_for_val = len(ts)

            bout_length = round(ts_end - ts_start, Parameters.TIMESTAMP_ROUND_VALUE)
            motion_index_slice = binary_df.iloc[index_start : index_end][common.INPUT_COL1_MI]
            sum_mi = sum(motion_index_slice)
            cnt_mi = len(motion_index_slice)
            data_slice = data[ts_index_start_for_val : ts_index_end_for_val]
            sum_data = sum(data_slice)
            cnt_data = len(data_slice)
            if cnt_data == 0:
                print("\nindex start ", index_start, "index_end ", index_end, "length ", index_end - index_start + 1)
                print("ts index start: ", ts_index_start_for_val, " end: ", ts_index_end_for_val)
            elif cur_binary_value == 0:
                if is_inside:
                    auc_0s_sum += sum_data
                    auc_0s_cnt += cnt_data
                    mi_0s_sum += sum_mi
                    mi_0s_cnt += cnt_mi
                    out_df_0s.loc[len(out_df_0s.index)] = [ts_start,
                                                        bout_length,
                                                        sum_mi/cnt_mi,
                                                        sum_data,
                                                        sum_data/cnt_data]
                else:
                    auc_0s_sum_out += sum_data
                    auc_0s_cnt_out += cnt_data
                    mi_0s_sum_out += sum_mi
                    mi_0s_cnt_out += cnt_mi
                    out_df_0s_out.loc[len(out_df_0s_out.index)] = [ts_start,
                                                        bout_length,
                                                        sum_mi/cnt_mi,
                                                        sum_data,
                                                        sum_data/cnt_data]
            else:
                if is_inside:
                    auc_1s_sum += sum_data
                    auc_1s_cnt += cnt_data
                    mi_1s_sum += sum_mi
                    mi_1s_cnt += cnt_mi
                    out_df_1s.loc[len(out_df_1s.index)] = [ts_start,
                                                        bout_length,
                                                        sum_mi/cnt_mi,
                                                        sum_data,
                                                        sum_data/cnt_data]
                else:
                    auc_1s_sum_out += sum_data
                    auc_1s_cnt_out += cnt_data
                    mi_1s_sum_out += sum_mi
                    mi_1s_cnt_out += cnt_mi
                    out_df_1s_out.loc[len(out_df_1s_out.index)] = [ts_start,
                                                        bout_length,
                                                        sum_mi/cnt_mi,
                                                        sum_data,
                                                        sum_data/cnt_data]

        # Reset the index to indicate the start of a new dataset.
        index_start = -1

    return True, [auc_0s_sum, auc_0s_cnt, out_df_0s,
                  auc_1s_sum, auc_1s_cnt, out_df_1s,
                  auc_0s_sum_out, auc_0s_cnt_out, out_df_0s_out,
                  auc_1s_sum_out, auc_1s_cnt_out, out_df_1s_out]


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
            msg = self.format(record)
            print(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def print_help():
    """
    Display help
    """

    print("\nHelp/Usage:\n")
    print(
        "python ff.py -i <input folder or .csv file> -o <output_directory> -v -h -c\n"
    )
    print("where:")
    print("-i (required): input folder or .csv file.")
    print("-o (optional): output folder.")
    print("-v (optional): run in verbose mode")
    print("-h (optional): print this help")
    print("-c (optional): run in console (no UI) mode")
    print("\nExamples:\n")
    print("\nProcess input file:")
    print("\tpython ff.py -i input.csv")
    print("\nProcess all the csv files from the input folder:")
    print("\tpython ff.py -i c:\\data\\input")
    print(
        "\nProcess all the csv files from the input folder and use the output folder:"
    )
    print("\tpython ff.py -i c:\\data\\input -o c:\\data\\output")
    print("\nNotes:")
    print("\tClose the output file prior to running.")
    sys.exit()

"""
------------------------------------------------------------
                UI related stuff
------------------------------------------------------------
"""

INPUT_FOLDER_NAME_BOX_MAX_WIDTH = 26
def refresh_param_names_combo_box(parameter_obj):
    param_name_list = parameter_obj.get_param_name_list()
    param_names_combo_box.clear()
    for param_name in param_name_list:
        param_names_combo_box.append(param_name)

    param_names_combo_box.show()

def select_input_dir(parameter_obj):
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
    param_name_list = parameter_obj.get_param_name_list()
    refresh_param_names_combo_box(parameter_obj)
    if len(param_name_list):
        param_window_duration, param_start_timestamp_series = parameter_obj.get_param_values(
            parameter_obj.get_currently_selected_param())
        refresh_param_values_ui(param_window_duration, param_start_timestamp_series)

def ui_process_cmd(parameter_obj):
    if not common.get_input_dir():
        app.warn(
            "Uh oh!", "No input folder specified. Please select an input folder and run again!")
        return

    """
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
    """
    main(common.get_input_dir(), parameter_obj)

def open_params_file(parameter_obj):
    param_file = parameter_obj.get_param_file_from_name(parameter_obj.get_currently_selected_param())
    if not os.path.isfile(param_file):
        app.warn(
            "Uh oh!", "Parameters file " + param_file + " is not a file!")
        return

    common.open_file(param_file)

def parse_cur_param_file(parameter_obj):
    """
    Parse the paramter file
    """

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

def set_time_window_duration_box_value(param_window_duration):
    time_window_duration_box.value = param_window_duration

def refresh_ts_list_box(ts_series):
    ts_series_list_box.clear()
    for val in ts_series:
        ts_series_list_box.append(val)

def refresh_param_values_ui(param_window_duration, param_start_timestamp_series):
    set_time_window_duration_box_value(param_window_duration)
    refresh_ts_list_box(param_start_timestamp_series)

def select_param(selected_param_value):
    """
    This method is called when the user selects a parameter from the parameter
    name drop down box.
    """
    global parameter_obj

    if not selected_param_value:
        return

    cur_selected_param = selected_param_value
    try:
        parameter_obj.set_currently_selected_param(selected_param_value)
    except:
        common.logger.error("Parameter name(%s) not found in list; unexpected")

    param_window_duration, param_start_timestamp_series = parameter_obj.get_param_values(cur_selected_param)
    refresh_param_values_ui(param_window_duration, param_start_timestamp_series)


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

    try:
        opts, args = getopt.getopt(argv, "i:vhco:")
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
        elif opt in ("-c"):
            console_mode = True
        else:
            print_help()

    if console_mode:
        main(input_dir, parameter_obj)
        sys.exit()

    # Main app
    app = App("", height=600, width=900)

    # App name box
    title = Text(app, text="Z-Score Splitting App",
                 size=16, font="Arial Bold", width=30)
    title.bg = "white"
    line()

    # Select input folder button
    Text(app, "Select Input Folder --> Check Parameters --> Process",
         font="Verdana bold")

    line()
    input_dir_button = PushButton(
        app, command=select_input_dir, args=[parameter_obj], text="Input Folder", width=26)
    input_dir_button.tk.config(font=("Verdana bold", 14))
    # Box to display the input folder
    line()
    input_dir_text_box = TextBox(app)
    # Non editable
    input_dir_text_box.disable()
    input_dir_text_box.width = INPUT_FOLDER_NAME_BOX_MAX_WIDTH
    input_dir_text_box.font = "Verdana bold"
    input_dir_text_box.text_size = 14
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
        param_box, pd.Series(dtype=np.float64), scrollbar=True, grid=[1, cnt],
        align="left", width=80, height=125)
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
    process_button = PushButton(app, text="Process", command=ui_process_cmd,
                                args=[parameter_obj], width=26)
    process_button.tk.config(font=("Verdana bold", 14))

    # Display the app
    app.display()

    logging.shutdown()