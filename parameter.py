#!/usr/bin/python

import pandas as pd
import os
import numpy as np
import glob
import unittest

CSV_EXT = ".csv"

class Parameters:
    PARAM_TIME_WINDOW_START_LIST = "Start_Timestamp_List"
    PARAM_TIME_WINDOW_DURATION = "Window_Duration_In_Sec"
    PARAMETERS_DIR_NAME = "parameters"
    TIME_DURATION_PARAMETER_FILE = "min_time.txt"
    _param_col_names = [PARAM_TIME_WINDOW_START_LIST, PARAM_TIME_WINDOW_DURATION]

    def __init__(self):
        self.reset()

    def reset(self):
        self._cur_selected_param = ""
        self._param_name_list = []
        self._param_df_list = []
        self._param_window_duration = 0
        self._param_start_timestamp_series = pd.Series(dtype=np.float64)
        self._param_dir = ""
        self._min_time_duration_before = 0
        self._min_time_duration_after = 0
        self._parse_file_exists = False

    def _set_param_dir(self, input_dir):
        self._param_dir = Parameters.get_param_dir(input_dir)
        print("self param dir", self._param_dir)

    def parse(self, input_dir):
        self.reset()
        self._param_dir = Parameters.get_param_dir(input_dir)
        param_dir = self._param_dir
        if not os.path.isdir(param_dir):
            raise ValueError("Parameter folder %s does not exist!", param_dir)

        search_path = os.path.join(param_dir, "*.csv")
        for param_file in glob.glob(search_path):
            if not os.path.isfile(param_file):
                continue

            param_file_name_without_ext = os.path.splitext(os.path.basename(param_file))[0]
            self._param_name_list.append(param_file_name_without_ext)
            param_df = pd.read_csv(param_file, names=Parameters.get_param_column_names(), header=None, skiprows=1)
            self._param_df_list.append(param_df)

        if len(self._param_name_list) > 0:
            self._cur_selected_param = self._param_name_list[0]

        self.parse_min_time_duration()

        return

    def get_currently_selected_param(self):
        return self._cur_selected_param

    def set_currently_selected_param(self, param_name):
        if param_name in self._param_name_list:
            self._cur_selected_param = param_name
        else:
            raise ValueError("Parameter name %s is not part of valid parameter list!", param_name)

    def get_param_name_list(self):
        return self._param_name_list

    def get_param_values(self, param_name):
        if not param_name:
            return self.get_default_parameter_values()

        try:
            param_index = self._param_name_list.index(param_name)
        except ValueError:
            raise ValueError("Parameter %s is not in the parameter list.", param_name)

        param_df = self._param_df_list[param_index]
        return Parameters.parse_param_df(param_df)

    def get_param_df(self):
        return self._param_df_list

    def get_param_df_for_param(self, param_name):
        try:
            param_index = self._param_name_list.index(param_name)
        except ValueError:
            raise ValueError("Parameter %s is not in the parameter list.", param_name)

        return self._param_df_list[param_index]

    def get_param_file_from_name(self, param_name):
        return os.path.join(self._param_dir, param_name + CSV_EXT)

    def set_param_value(self, param_name, param_df):
        self._param_name_list.append(param_name)
        self._param_df_list.append(param_df)

    def get_default_parameter_values(self):
        # Window duration, time series
        return 0, pd.Series(dtype=np.float64)

    def _write_params(self):
        if not os.path.isdir(self._param_dir):
            raise ValueError("Parameter folder %s does not exist!", self._param_dir)

        for index, param_name in enumerate(self._param_name_list):
            param_file_name = os.path.join(self._param_dir, param_name + ".csv")
            param_df = self._param_df_list[index]
            param_df.to_csv(param_file_name, index=False, header=True)

        return

    def get_min_time_duration_file(self):
        return os.path.join(self._param_dir, Parameters.TIME_DURATION_PARAMETER_FILE)

    def parse_min_time_duration(self):
        itr = 0
        param_min_t_file = self.get_min_time_duration_file()
        try:
            with open(param_min_t_file) as min_t_file:
                self._param_file_exists = True
                while True:
                    line = min_t_file.readline().rstrip()
                    if not line:
                        break
                    try:
                        t_duration = float(line)
                        if itr == 0:
                            self._min_time_duration_before = t_duration
                            itr += 1
                        else:
                            self._min_time_duration_after = t_duration
                            break
                    #TODO: Can en except raise an exception?
                    except ValueError:
                        raise ValueError(
                            "Min time duration(%s) from file(%s) cannot be converted to a "
                            "number. Using default of %d", line, param_min_t_file, t_duration)
                min_t_file.close()
        except IOError:
            self._param_file_exists = False
            pass

    def get_min_time_duration_values(self):
        return self._param_file_exists, self._min_time_duration_before , self._min_time_duration_after

    def set_min_time_duration_values(self, min_time_duration_befor, min_time_duration_after):
        param_min_t_file = self.get_min_time_duration_file()
        try:
            # "w+" will create the file if not exist.
            with open(param_min_t_file, "w+") as min_t_file:
                min_t_file.write(min_time_duration_befor)
                min_t_file.write("\n")
                min_t_file.write(min_time_duration_after)
                self._param_file_exists = True
                min_t_file.close()
                self._min_time_duration_before = min_time_duration_befor
                self._min_time_duration_after = min_time_duration_after
        except IOError:
            raise ValueError("Min time duration file(%s) cannot be created or written to.", param_min_t_file)
            pass

    @staticmethod
    def get_param_column_names():
        return Parameters._param_col_names

    @staticmethod
    def parse_param_df(df):
        value = df[Parameters.PARAM_TIME_WINDOW_DURATION].iat[0]
        w_duration = 0
        if not pd.isnull(value):
            w_duration = value

        ts_series = df[Parameters.PARAM_TIME_WINDOW_START_LIST]
        ts_series.sort_values(ascending=True)

        return w_duration, ts_series

    @staticmethod
    def get_param_dir(input_dir):
        return os.path.join(input_dir, Parameters.PARAMETERS_DIR_NAME)

"""
------------------------------------------------------------
                Unit Tests
------------------------------------------------------------
# Run these tests using `python -m unittest parameter`
------------------------------------------------------------
"""
param1 = {
    Parameters.PARAM_TIME_WINDOW_START_LIST:    [100,   200,    300,    400,    500],
    Parameters.PARAM_TIME_WINDOW_DURATION:      [30,    np.nan, np.nan, np.nan, np.nan]
}

class ParameterTest(unittest.TestCase):
    TEST_DATA_DIR = "test_data"
    TEST_TRASH_DIR = "trash"
    def setUp(self):
        self.param = Parameters()
        # TODO: Create test trash dir with Parameters and cleanup

    def validate_df(self, df, expected_data):
        expected_df = pd.DataFrame(expected_data)
        self.assertEqual(expected_df.equals(df), True)

    def test_param_bvt(self):
        param = Parameters()
        input_dir = ParameterTest.get_trash_dir()
        param._set_param_dir(input_dir)
        expected_df = pd.DataFrame(param1)
        param.set_param_value("param1", expected_df)
        param._write_params()
        param.parse(input_dir)
        param_list = param.get_param_name_list()
        expected_param_list = ["param1"]
        self.assertEqual(param_list == expected_param_list, True)
        param_df = param.get_param_df_for_param(param_list[0])
        self.assertEqual(expected_df.equals(param_df), True)

    @staticmethod
    def get_test_dir():
        return os.path.join(os.getcwd(), ParameterTest.TEST_DATA_DIR)

    @staticmethod
    def get_trash_dir():
        return os.path.join(ParameterTest.get_test_dir(), ParameterTest.TEST_TRASH_DIR)