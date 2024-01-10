#!/usr/bin/python

import pandas as pd
import os
import numpy as np
from numpy import NAN as NAN
import glob
import unittest
import shutil
from pathlib import Path

CSV_EXT = ".csv"

class Parameters:
    """
    A class used to represent and parse Parameters.

    ...

    Attributes
    ----------
    _cur_selected_param : str
        the currently selected parameter
    _param_name_list : str
        the list of parameters
    _param_df_list : str
        the list of parameter dataframe, one for each parameter
    _param_window_duration : float
        the parameter window duration (default 0)
    _param_dir : str
        the parameter directory
    _min_time_duration_before : float
        the minimum duration time before (default 0)
    _min_time_duration_after : float
        the minimum duration time before (default 0)
    _param_file_exists: bool
        indicates whether the parameter file exists
    """

    PARAM_TIME_WINDOW_START_LIST = "Start_Timestamp_List"
    PARAM_TIME_WINDOW_DURATION = "Window_Duration_In_Sec"
    PARAMETERS_DIR_NAME = "parameters"
    TIME_DURATION_PARAMETER_FILE = "min_time.txt"
    MIN_TIME_DURATION_BEFORE_DEFAULT = float(0)
    MIN_TIME_DURATION_AFTER_DEFAULT = float(0)
    PARAM_WINDOW_DURATION_DEFAULT = 0
    _param_col_names = [PARAM_TIME_WINDOW_START_LIST, PARAM_TIME_WINDOW_DURATION]

    def __init__(self):
        self.reset()

    def reset(self):
        """resets the object to the default values"""
        self._cur_selected_param = ""
        self._param_name_list = []
        self._param_df_list = []
        self._param_window_duration = Parameters.PARAM_WINDOW_DURATION_DEFAULT
        self._param_dir = ""
        self._min_time_duration_before = Parameters.MIN_TIME_DURATION_BEFORE_DEFAULT
        self._min_time_duration_after = Parameters.MIN_TIME_DURATION_AFTER_DEFAULT
        self._param_file_exists = False

    def parse(self, input_dir: str):
        """parse the input directory and populate the parameter values with the parsed values

        Parameters
        ----------
        input_dir : str
            The path of the input directory

        Raises
        ------
        ValueError
            If the parameter directory does not exist.
        """
        self.reset()
        self._param_dir = Parameters.get_param_dir(input_dir)
        param_dir = self._param_dir
        if not os.path.isdir(param_dir):
            raise ValueError("Parameter folder", param_dir, "does not exist!")

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

        self._parse_min_time_duration()

        return

    def get_currently_selected_param(self):
        """get the currently selected parameter"""
        return self._cur_selected_param

    def set_currently_selected_param(self, param_name: str):
        """set the currently selected paramter to the provided value

        Parameters
        ----------
        param_name : str
            The name of the parameter

        Raises
        ------
        ValueError
            If the parameter is not part of the parameter list.
        """
        if param_name in self._param_name_list:
            self._cur_selected_param = param_name
        else:
            raise ValueError("Parameter", param_name, "is not part of valid parameter list!")

    def get_param_name_list(self):
        """get the list of parameter names"""
        return self._param_name_list


    def get_param_values(self, param_name: str) -> (float, list):
        """get the parameter values for the given parameter

        Parameters
        ----------
        param_name : str
            The name of the parameter

        Raises
        ------
        ValueError
            If the parameter is not part of the current parameter list.
        """
        if param_name == "":
            return self.get_default_parameter_values()
        param_df = self.get_param_df_for_param(param_name)

        return Parameters.parse_param_df(param_df)

    def set_param_value(self, param_name: str, param_df: pd.DataFrame):
        """set the parameter value for the given parameter to the provided value.

        Parameters
        ----------
        param_name : str
            The name of the parameter

        param_df: pd.Dataframe
            The dataframe value for this parameter.
        """
        self._param_name_list.append(param_name)
        self._param_df_list.append(param_df)

    def set_time_window_duration(self, time_window: float):
        """set the parameter value for the time window duration to the provided value.

        Parameters
        ----------
        time_window : float
            The time window duration value to set
        """
        self.self._param_window_duration = time_window

    def get_param_df(self):
        """get all the parameter dataframe values for all of the parameters"""

        return self._param_df_list

    def get_param_df_for_param(self, param_name: str) -> pd.DataFrame:
        """get the parameter dataframe values for the given parameter

        Parameters
        ----------
        param_name : str
            The name of the parameter

        Raises
        ------
        ValueError
            If the parameter is not part of the current parameter list.
        """

        try:
            param_index = self._param_name_list.index(param_name)
        except ValueError:
            raise ValueError("Parameter", param_name, "is not in the parameter list.")

        return self._param_df_list[param_index]

    def get_param_file_from_name(self, param_name: str) -> str:
        """get the parameter dataframe values for the given parameter

        Parameters
        ----------
        param_name : str
            The name of the parameter

        Returns
        ----------
        Returns the parameter file path for the given parameter
        """
        return os.path.join(self._param_dir, param_name + CSV_EXT)

    def get_default_parameter_values(self) -> (float, pd.Series):
        """get the default parameter values

        Returns
        ----------
        Returns the default time window duration and the default timeseries dataframe
        """

        return Parameters.PARAM_WINDOW_DURATION_DEFAULT, pd.Series(dtype=np.float64)

    def get_min_time_duration_file(self) -> str:
        """get the path to the min time duration file


        Returns
        ----------
        Returns the path to the min time duration file
        """

        return os.path.join(self._param_dir, Parameters.TIME_DURATION_PARAMETER_FILE)

    def get_min_time_duration_values(self) -> (bool, float, float):
        """get the path to the min time duration file

        Returns
        ----------
        Returns a bool indicating whether the parameter file exists, the min
        time duration before and min time duration after
        """

        return self._param_file_exists, float(self._min_time_duration_before), float(self._min_time_duration_after)

    def set_min_time_duration_values(self,
                                     min_time_duration_before: float,
                                     min_time_duration_after: float):
        """set the parameter dataframe values for the given parameter and also
        write the values to the min time duration paraemeter file.

        Parameters
        ----------
        min_time_duration_before : float
            The min time duration before value

        min_time_duration_after : float
            The min time duration after value

        Raises
        ------
        ValueError
            If an error occurs while writing to the min time duration parameter file.
        """

        param_min_t_file = self.get_min_time_duration_file()
        try:
            # "w+" will create the file if not exist.
            with open(param_min_t_file, "w+") as min_t_file:
                min_t_file.write(str(min_time_duration_before))
                min_t_file.write("\n")
                min_t_file.write(str(min_time_duration_after))
                self._param_file_exists = True
                min_t_file.close()
                self._min_time_duration_before = min_time_duration_before
                self._min_time_duration_after = min_time_duration_after
        except IOError:
            raise ValueError("Min time duration file", param_min_t_file, "cannot be created or written to.")
            pass

    def get_ts_series_for_timestamps(self,
                                     param_name: str,
                                     ts_start: float,
                                     ts_end: float):
        """get the timestamp series for the given parameter name and the timestamp duration.

        Parameters
        ----------
        param_name : str
            The parameter name to which to apply the timestamp for

        ts_start : float
            The start timestamp

        ts_end : float
            The end timestamp

        Raises
        ------
        ValueError
            If an error occurs while writing to the min time duration parameter file.

        Returns
        ----------
        Returns a split timestamp series for the given parameter and timestamp duration with an
        indication of whether the split timestamp fits within the window or outside. This is
        best explained with an example.
        For example, if the timestamp series for this parameter is
        Window duration: 5s
        Timestamps: 10, 20, 30, 40
        So, for a given timestamp of [8, 23] (i.e from 8 to 17 seconds), this routine will return:
        [[10, 15, True], [15, 20, False], [20, 23, True]]

        Another example:
        Window duration: 5s
        Timestamps: 10, 20
        So, for a given timestamp of [5, 35] (i.e from 5 to 35 seconds), this routine will return:
        [[5, 10, False], [10, 15, True], [15, 20, False], [20, 25, True], [25, 35, False]]
        """
        ts_split = []
        param_window_duration, ts = self.get_param_values(param_name)
        #print("param name: ", param_name, "ts_start: ", ts_start, "ts_end: ", ts_end, "window dur: ", param_window_duration)
        indices = list(filter(lambda x: (ts[x] >= ts_start and ts[x] < ts_end) or
                              ((ts[x] < ts_start) and (ts[x] + param_window_duration) > ts_start), range(len(ts))))
        # No timestamp in the series fits within the provided time. Mark
        # the whole duration as outside.
        if len(indices) == 0:
            # For empty parameter, everything is considered to be in range.
            if param_name == "":
                ts_split.append([ts_start, ts_end, True])
            else:
                ts_split.append([ts_start, ts_end, False])
            return ts_split

        start = ts_start
        idx = 0
        while True:
            #print("start: ", start, " ts[idx]: ", ts[indices[idx]])
            if start < ts[indices[idx]]:
                is_in = False
                end = min(ts_end, ts[indices[idx]])
            else:
                is_in = True
                end = min(ts_end, ts[indices[idx]] + param_window_duration)
                idx += 1

            ts_split.append([start, end, is_in])

            # If we have reached the end of the ts series and there is
            # still some left in the duration, just add the rest.
            if idx >= len(indices) and end < ts_end:
                ts_split.append([end, ts_end, False])
                end = ts_end

            if end == ts_end:
                break

            start = end

        return ts_split

    @staticmethod
    def get_param_column_names() -> list:
        """get the parameter column names

        Returns
        ----------
        Returns the parsed parameter column names.
        """

        return Parameters._param_col_names

    @staticmethod
    def parse_param_df(df) -> (float, list):
        """Parses the dataframe into the window duration and the time series.

        Parameters
        ----------
        df : pandas.dataframe
            The dataframe to parse.

        Returns
        ----------
        Returns the parsed time window duration and time series.
        """
        value = df[Parameters.PARAM_TIME_WINDOW_DURATION].iat[0]
        w_duration = 0
        if not pd.isnull(value):
            w_duration = value

        ts_series = df[Parameters.PARAM_TIME_WINDOW_START_LIST]
        ts_series.sort_values(ascending=True)

        return w_duration, ts_series

    @staticmethod
    def get_param_dir(input_dir: str) -> str:
        """Returns the parameter directory string for the given input dir.

        Parameters
        ----------
        df : str
            The input dir path.

        Returns
        ----------
        Returns the parameter directory string for the given input dir.
        """

        return os.path.join(input_dir, Parameters.PARAMETERS_DIR_NAME)

    def _set_param_dir(self, input_dir: str):
        """Set the parameter directory for the given input dir.

        Parameters
        ----------
        input_dir : str
            The input dir path.
        """

        self._param_dir = Parameters.get_param_dir(input_dir)

    def _parse_min_time_duration(self):
        """Parse the min time duration file.

        Raises
        ------
        ValueError
            If an error occurs while trying to parse the values from the min time
            duration file.
        """

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
                    except ValueError:
                        raise ValueError(
                            "Min time duration", line, "from file", param_min_t_file,
                            "cannot be converted to a number. Using default of", t_duration)
                min_t_file.close()
        except IOError:
            self._param_file_exists = False
            pass

    def _write_params(self):
        """Write the parameters out to the respective parameter files.

        Raises
        ------
        ValueError
            If an parameter dir is not a directory.
        """

        if not os.path.isdir(self._param_dir):
            raise ValueError("Parameter folder", self._param_dir, "does not exist!", )

        for index, param_name in enumerate(self._param_name_list):
            param_file_name = self._get_file_name_for_param(param_name)
            param_df = self._param_df_list[index]
            param_df.to_csv(param_file_name, index=False, header=True)

        return

    def _get_file_name_for_param(self, param_name: str) -> str:
        """Set the parameter directory for the given input dir.

        Parameters
        ----------
        param_name : str
            The parameter name.

        Returns
        ----------
        Returns the file name corresponding to the given parameter name.
        """

        return os.path.join(self._param_dir, param_name + ".csv")

"""
------------------------------------------------------------
                Unit Tests
------------------------------------------------------------
# Run these tests using `python -m unittest parameter`
------------------------------------------------------------
"""
param1 = {
    Parameters.PARAM_TIME_WINDOW_START_LIST:    [100, 200, 300, 400, 500],
    Parameters.PARAM_TIME_WINDOW_DURATION:      [30, np.nan, np.nan, np.nan, np.nan]
}

param2 = {
    Parameters.PARAM_TIME_WINDOW_START_LIST:    [1.0, 2, 3, 4.0, 5],
    Parameters.PARAM_TIME_WINDOW_DURATION:      [30, np.nan, np.nan, np.nan, np.nan]
}

min_time_duration_validation_set = [
    [5, 10],
    [1.0, 2],
    [20.0, 30.0],
    [0, 0]
]

class ParameterTest(unittest.TestCase):
    TEST_DATA_DIR = "test_data"
    TEST_TRASH_DIR = "trash"
    def setUp(self):
        self.param = Parameters()
        # TODO: Create test trash dir with Parameters and cleanup

    def validate_df(self, df, expected_data):
        expected_df = pd.DataFrame(expected_data)
        self.assertEqual(expected_df.equals(df), True)

    def reset(self) -> str:
        input_dir = ParameterTest.get_trash_dir()
        param_dir = Parameters.get_param_dir(input_dir)
        shutil.rmtree(param_dir)
        path = Path(param_dir)
        path.mkdir(parents=True, exist_ok=True)

        return input_dir

    def validate_min_t_duration(self, input_dir):
        expected_param = Parameters()
        expected_param._set_param_dir(input_dir)

        # Min time duration values should be set to default valuse without any
        # min time duration parameter present.
        (param_file_exists,
         min_time_duration_before,
         min_time_duration_after) = expected_param.get_min_time_duration_values()
        self.assertEqual(param_file_exists == False, True)
        self.assertEqual(min_time_duration_before == Parameters.MIN_TIME_DURATION_BEFORE_DEFAULT,
                         True)
        self.assertEqual(min_time_duration_after == Parameters.MIN_TIME_DURATION_AFTER_DEFAULT,
                         True)

        # Set some min time duration values, parse and make sure they match
        for row in min_time_duration_validation_set:
            validate_param = Parameters()
            expected_param.set_min_time_duration_values(row[0], row[1])
            validate_param.parse(input_dir)
            (param_file_exists,
            min_time_duration_before,
            min_time_duration_after) = expected_param.get_min_time_duration_values()
            self.assertEqual(param_file_exists == True, True)
            self.assertEqual(min_time_duration_before == row[0], True)
            self.assertEqual(min_time_duration_after == row[1], True)

    def test_param_bvt(self):
        input_dir = self.reset()
        expected_param = Parameters()
        expected_param._set_param_dir(input_dir)

        # Basic test of writing a parameter with values and then validate that
        # the parsed values match.
        PARAM1_NAME = "param1"
        expected_df1 = pd.DataFrame(param1)
        expected_param.set_param_value(PARAM1_NAME, expected_df1)
        expected_param._write_params()
        validate_param = Parameters()
        validate_param.parse(input_dir)
        param_list = validate_param.get_param_name_list()
        expected_param_list = [PARAM1_NAME]
        self.assertEqual(param_list == expected_param_list, True)
        param_df1 = validate_param.get_param_df_for_param(PARAM1_NAME)
        self.assertEqual(expected_df1.equals(param_df1), True)

        # Add more parameters.
        PARAM2_NAME = "param2"
        expected_df2 = pd.DataFrame(param2)
        expected_param.set_param_value(PARAM2_NAME, expected_df2)
        expected_param._write_params()
        validate_param = Parameters()
        validate_param.parse(input_dir)
        param_list = validate_param.get_param_name_list()
        expected_param_list = [PARAM1_NAME, PARAM2_NAME]
        self.assertEqual(param_list == expected_param_list, True)
        param_df2 = validate_param.get_param_df_for_param(PARAM2_NAME)
        self.assertEqual(expected_df2.equals(param_df2), True)

        # Test for set/get_currently_selected_param
        expected_param.set_currently_selected_param(PARAM2_NAME)
        cur_selected_param = expected_param.get_currently_selected_param()
        self.assertEqual(cur_selected_param == PARAM2_NAME, True)

        # Validate min time duration values after the above parameters
        # have been set on the input dir.
        self.validate_min_t_duration(input_dir)

    def test_min_t_param(self):
        input_dir = self.reset()
        expected_param = Parameters()
        expected_param._set_param_dir(input_dir)

        # Validate that min time duration parameters can work without
        # any other parameters present.
        self.validate_min_t_duration(input_dir)

    def test_get_ts_series_for_timestamps(self):
        param_val = {
            Parameters.PARAM_TIME_WINDOW_START_LIST:    [10, 20, 30],
            Parameters.PARAM_TIME_WINDOW_DURATION:      [5, np.nan, np.nan]
        }
        ts1 = [5, 8]
        expected_out1 = [[5, 8, False]]

        ts2 = [5, 37]
        expected_out2 = [[5, 10, False], [10, 15, True], [15, 20, False],
                        [20, 25, True], [25, 30, False], [30, 35, True],
                        [35, 37, False]]

        ts3 = [5, 23]
        expected_out3 = [[5, 10, False], [10, 15, True], [15, 20, False],
                        [20, 23, True]]

        ts4 = [5, 13]
        expected_out4 = [[5, 10, False], [10, 13, True]]

        ts5 = [20, 25]
        expected_out5 = [[20, 25, True]]

        ts6 = [10, 15]
        expected_out6 = [[10, 15, True]]

        ts7 = [30, 35]
        expected_out7 = [[30, 35, True]]

        ts8 = [30, 43]
        expected_out8 = [[30, 35, True], [35, 43, False]]

        ts9 = [45, 70]
        expected_out9 = [[45, 70, False]]

        ts10 = [25, 30]
        expected_out10 = [[25, 30, False]]

        expected_ts = [[ts1, expected_out1], [ts2, expected_out2],
                       [ts3, expected_out3], [ts4, expected_out4],
                       [ts5, expected_out5], [ts6, expected_out6],
                       [ts7, expected_out7], [ts8, expected_out8],
                       [ts9, expected_out9], [ts10, expected_out10]]
        param = Parameters()
        PARAM_NAME = "param"
        df = pd.DataFrame(param_val)
        #print(param_val)
        for val in expected_ts:
            #print("timestamp duration: ", val[0])
            input_dir = self.reset()
            param._set_param_dir(input_dir)
            param.set_param_value(PARAM_NAME, df)
            ts_split = param.get_ts_series_for_timestamps(PARAM_NAME, val[0][0], val[0][1])
            #print("Timestamp splits: ", ts_split)
            self.assertEqual(ts_split == val[1], True)

    @staticmethod
    def get_test_dir():
        return os.path.join(os.getcwd(), ParameterTest.TEST_DATA_DIR)

    @staticmethod
    def get_trash_dir():
        return os.path.join(ParameterTest.get_test_dir(), ParameterTest.TEST_TRASH_DIR)

    @staticmethod
    def remove_file(file_path):
        if os.path.exists(file_path):
            os.remove(file_path)