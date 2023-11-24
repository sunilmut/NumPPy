import h5py
import sys
import getopt
import logging
import numpy as np
from common import *

OUTPUT_LOG_FILE = "output.txt"

def process(filename):
    print("opening file", filename)
    with h5py.File(filename, "r") as f:
        # Print all root level object names (aka keys) 
        # these can be group or dataset names 
        ls = list(f.keys())
        print("keys: ", ls)

        # get first object name/key; may or may NOT be a group
        #a_group_key = list(ls[0])

        # get the object type for a_group_key: usually group or dataset
        #print(type(f[a_group_key]))

        data = f.get(ls[0])
        data1 = np.array(data)
        print("shape of dataset1: \n", data1.shape)

        """
        # If a_group_key is a group name, 
        # this gets the object names in the group and returns as a list
        data = list(f[a_group_key])

        # If a_group_key is a dataset name, 
        # this gets the dataset values and returns as a list
        data = list(f[a_group_key])
        # preferred methods to get dataset values:
        ds_obj = f[a_group_key]      # returns as a h5py dataset object
        ds_arr = f[a_group_key][()]  # returns as a numpy array
        print(data[0:10])
        """

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
    logger = logging.getLogger(__name__)
    logger.addHandler(progress)
    argv = sys.argv[1:]
    console_mode = False
    separate_files = False
    output_folder = None

    try:
        opts, args = getopt.getopt(argv, "vhco:d:i:s")
    except getopt.GetoptError as e:
        logger.error("USAGE ERROR: %s", e)
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

    #process(input_dir)
    success, df = parse_input_file_into_df(input_dir, NUM_INITIAL_ROWS_TO_SKIP + 1)
    print(df)