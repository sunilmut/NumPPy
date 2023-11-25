import h5py
import sys
import getopt
import logging
import numpy as np
import common
import os
import pandas as pd
from pandas.api.types import is_integer_dtype

# constants
OUTPUT_LOG_FILE = "output.txt"

OUTPUT_COL0_TS = 'Start time (sec)'
OUTPUT_COL1_LEN = 'Bout length (sec)'
OUTPUT_COL2_MI_AVG = 'Motion Index Average'
OUTPUT_COL3_DATA_AUC = 'Area under curve (data)'
OUTPUT_COL4_DATA_AVG = 'Data Average'
OUTPUT_COLUMN_NAMES=[OUTPUT_COL0_TS, OUTPUT_COL1_LEN,
                     OUTPUT_COL2_MI_AVG, OUTPUT_COL3_DATA_AUC,
                     OUTPUT_COL4_DATA_AVG]

# function to read hdf5 file
def read_hdf5(event, filepath, key):
    if event:
        event = event.replace("\\","_")
        event = event.replace("/","_")
        op = os.path.join(filepath, event+'.hdf5')
    else:
        op = filepath

    if os.path.exists(op):
        with h5py.File(op, 'r') as f:
            arr = np.asarray(f[key])
    else:
        common.logger.error(f"{op}.hdf5 file does not exist")
        raise Exception('{}.hdf5 file does not exist'.op)

    return arr

def process(filename, ts_file, csv_file):
    # Get the timeshift and binary timestamps.
    timeshift_val, num_rows_processed = common.get_timeshift_from_input_file(csv_file)
    print("timeshift value is: ", timeshift_val)

    success, binary_df = common.parse_input_file_into_df(csv_file,
                                                  common.NUM_INITIAL_ROWS_TO_SKIP + num_rows_processed)
    if not success:
        common.log.error("Unable to parse the binary file")
        return

    # Get the list of data values.
    data = read_hdf5('', filename, 'data')

    # Get the timestamps for the data values
    print("opening file: ", ts_file)
    ts = read_hdf5('', ts_file, 'timestampNew')

    # Perform some basic checks on the data sets.
    if len(ts) != len(data):
        common.logger.error("Timestamp series length(%d) does not match data series length(%d)",
                            len(ts),
                            len(data))
        return
    
    if not binary_df[common.INPUT_COL0_TS].is_monotonic:
        common.logger.error("Binary timestamp values are not sorted.")
        return
    
    # Make sure the 'binary' column is actually binary.
    if not (
        is_integer_dtype(binary_df[common.INPUT_COL2_FREEZE])
        and binary_df[common.INPUT_COL2_FREEZE].min() == 0
        and binary_df[common.INPUT_COL2_FREEZE].max() == 1
    ):
        common.logger.error("Binary column contains non-binary data.")
        return

    # Timestamp series should be sorted.        
    if not np.all(np.diff(ts) >= 0):
        common.logger.error("Timestamp series is not sorted.")
        return
    
    index_start = -1
    row_count = binary_df.shape[0]
    auc_0s_sum = 0
    auc_0s_cnt = 0
    mi_0s_sum = 0
    mi_0s_cnt = 0
    out_df_0s = pd.DataFrame(columns=OUTPUT_COLUMN_NAMES)
    auc_1s_sum = 0
    auc_1s_cnt = 0
    mi_1s_sum = 0
    mi_1s_cnt = 0
    out_df_1s = pd.DataFrame(columns=OUTPUT_COLUMN_NAMES)
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

        #print("\nindex start ", index_start, "index_end ", index_end, "length ", index_end - index_start + 1)
        ts_start = binary_df.iloc[index_start][common.INPUT_COL0_TS] + timeshift_val
        ts_end = binary_df.iloc[index_end][common.INPUT_COL0_TS] + timeshift_val
        #print(ts_start, " - ", ts_end)
        ts_index_start_for_val = np.argmax(ts >= ts_start)
        ts_index_end_for_val = np.argmax(ts > ts_end)
        #print("ts index start: ", ts_index_start_for_val, " end: ", ts_index_end_for_val)
        bout_length = ts_end - ts_start
        motion_index_slice = binary_df.iloc[index_start : index_end + 1][common.INPUT_COL1_MI]
        sum_mi = sum(motion_index_slice)
        cnt_mi = len(motion_index_slice)
        #print(motion_index_slice)
        data_slice = data[ts_index_start_for_val : ts_index_end_for_val + 1]
        sum_data = sum(data_slice)
        cnt_data = len(data_slice)
        #print(data_slice)
        #print("sum: ", sum_data, " count: ", cnt_data)
        if cur_binary_value == 0:
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
            auc_1s_sum += sum_data
            auc_1s_cnt += cnt_data
            mi_1s_sum += sum_mi
            mi_1s_cnt += cnt_mi
            out_df_1s.loc[len(out_df_0s.index)] = [ts_start,
                                                   bout_length,
                                                   sum_mi/cnt_mi,
                                                   sum_data,
                                                   sum_data/cnt_data]

        # Reset the index to indicate the start of a new dataset.
        index_start = -1

    auc_0s_avg = auc_0s_sum/auc_0s_cnt
    auc_1s_avg = auc_1s_sum/auc_1s_cnt
    print(out_df_0s)
    print(out_df_1s)


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
        opts, args = getopt.getopt(argv, "i:t:s:vhcdo:")
    except getopt.GetoptError as e:
        common.logger.error("USAGE ERROR: %s", e)
        print_help()
    for opt, arg in opts:
        if opt == "-h":
            print_help()
        elif opt in ("-i"):
            input_dir = arg
        elif opt in ("-t"):
            ts_file = arg
        elif opt in ("-s"):
            csv_file = arg
        elif opt in ("-o"):
            output_folder = arg
        elif opt in ("-v"):
            logging.basicConfig(level=logging.DEBUG)
        elif opt in ("-c"):
            console_mode = True
        else:
            print_help()

    process(input_dir, ts_file, csv_file)