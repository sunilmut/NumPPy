#!/usr/bin/python

import pandas as pd
import os
import numpy as np
import glob

CSV_EXT = ".csv"

class Parameters:
    PARAM_TIME_WINDOW_START_LIST = "Start_Timestamp_List"
    PARAM_TIME_WINDOW_DURATION = "Window_Duration_In_Sec"
    PARAMETERS_DIR_NAME = "parameters"
    _param_col_names = [PARAM_TIME_WINDOW_START_LIST, PARAM_TIME_WINDOW_DURATION]

    def __init__(self):
        # currently selected parameter
        self._cur_selected_param = ""
        self._param_name_list = []
        # Parameter values as dataframe. There is one dataframe for each parameter
        self._param_df_list = []
        self._param_window_duration = 0
        self._param_start_timestamp_series = pd.Series(dtype=np.float64)
        self._param_dir = ""

    def reset(self):
        self._cur_selected_param = ""
        self._param_name_list = []
        self._param_df_list = []
        self._param_window_duration = 0
        self._param_start_timestamp_series = pd.Series(dtype=np.float64)
        self._param_dir = ""

    def parse(self, input_dir):
        self.reset()
        self._param_dir = os.path.join(input_dir, Parameters.PARAMETERS_DIR_NAME)
        param_dir = self._param_dir
        if not os.path.isdir(param_dir):
            raise ValueError("Parameter folder " + param_dir + " does not exist!")

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

        return

    def get_currently_selected_param(self):
        return self._cur_selected_param

    def set_currently_selected_param(self, param_name):
        if param_name in self._param_name_list:
            self._cur_selected_param = param_name
        else:
            raise ValueError("Parameter name " + param_name + " is not part of valid parameter list!")

    def get_param_name_list(self):
        return self._param_name_list

    def get_param_values(self, param_name):
        if not param_name:
            return self.get_default_parameter_values()

        try:
            param_index = self._param_name_list.index(param_name)
        except ValueError:
            raise ValueError("Parameter " + param_name + " is not in the parameter list.")

        param_df = self._param_df_list[param_index]
        return Parameters.parse_param_df(param_df)

    def get_param_df(self):
        return self._param_df_list

    def get_param_file_from_name(self, param_name):
        return os.path.join(self._param_dir, param_name + CSV_EXT)

    def get_default_parameter_values(self):
        # Window duration, time series
        return 0, pd.Series(dtype=np.float64)

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
    def get_param_column_names():
        return Parameters._param_col_names