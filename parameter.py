#!/usr/bin/python

import pandas as pd
import os
import numpy as np
from numpy import NAN as NAN
import glob
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
        the list of parameter DataFrame, one for each parameter
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
    PARAM_TIME_WINDOW_END_LIST = "End_Timestamp_List"
    PARAM_TIME_WINDOW_DURATION = "Window_Duration_In_Sec"
    PARAMETERS_DIR_NAME = "parameters"
    TIME_DURATION_PARAMETER_FILE = "min_time.txt"
    MIN_TIME_DURATION_BEFORE_DEFAULT = float(0)
    MIN_TIME_DURATION_AFTER_DEFAULT = float(0)
    PARAM_WINDOW_DURATION_DEFAULT = 0
    _param_col_names = [PARAM_TIME_WINDOW_START_LIST, PARAM_TIME_WINDOW_DURATION]
    TIMESTAMP_ROUND_VALUE = 6
    TIME_PRECISION = float(1) / pow(10, TIMESTAMP_ROUND_VALUE)

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

            param_file_name_without_ext = os.path.splitext(
                os.path.basename(param_file)
            )[0]
            self._param_name_list.append(param_file_name_without_ext)
            param_df = pd.read_csv(
                param_file,
                names=Parameters.get_param_column_names(),
                header=None,
                skiprows=1,
            )
            self._param_df_list.append(param_df)

        if len(self._param_name_list) > 0:
            self._cur_selected_param = self._param_name_list[0]

        self._parse_min_time_duration()

        return

    def get_currently_selected_param(self):
        """get the currently selected parameter"""
        return self._cur_selected_param

    def set_currently_selected_param(self, param_name: str):
        """set the currently selected parameter to the provided value

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
            raise ValueError(
                "Parameter", param_name, "is not part of valid parameter list!"
            )

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

    def get_param_values_as_series(self, param_name: str) -> (float, list):
        """get the parameter values for the given parameter as [start, end] series

        Parameters
        ----------
        param_name : str
            The name of the parameter

        Raises
        ------
        ValueError
            If the parameter is not part of the current parameter list.
        """

        param_window_duration, ts = self.get_param_values(param_name)
        ts_series = pd.DataFrame(
            columns=[
                Parameters.PARAM_TIME_WINDOW_START_LIST,
                Parameters.PARAM_TIME_WINDOW_END_LIST,
            ]
        )
        for start in ts:
            ts_series.loc[len(ts_series.index)] = [start, start + param_window_duration]

        return ts_series

    def get_combined_params_ts_series(self):
        param_list = self.get_param_name_list()
        ts_series_combined = pd.DataFrame(
            columns=[
                Parameters.PARAM_TIME_WINDOW_START_LIST,
                Parameters.PARAM_TIME_WINDOW_END_LIST,
            ]
        )
        for param in param_list:
            ts_series = self.get_param_values_as_series(param)
            ts_series_combined = pd.concat([ts_series_combined, ts_series])

        ts_series_combined.sort_values(
            by=Parameters.PARAM_TIME_WINDOW_START_LIST, ascending=True, inplace=True
        )

        ts_series_out = pd.DataFrame(
            columns=[
                Parameters.PARAM_TIME_WINDOW_START_LIST,
                Parameters.PARAM_TIME_WINDOW_END_LIST,
            ]
        )
        i = 0
        while i < len(ts_series_combined):
            j = i + 1
            start = ts_series_combined.iloc[i][Parameters.PARAM_TIME_WINDOW_START_LIST]
            end = ts_series_combined.iloc[i][Parameters.PARAM_TIME_WINDOW_END_LIST]
            while j < len(ts_series_combined):
                cur_start = ts_series_combined.iloc[j][
                    Parameters.PARAM_TIME_WINDOW_START_LIST
                ]
                cur_end = ts_series_combined.iloc[j][
                    Parameters.PARAM_TIME_WINDOW_END_LIST
                ]
                if cur_start > end:
                    break
                if cur_end > end:
                    end = cur_end
                j += 1
            i = j
            ts_series_out.loc[len(ts_series_out.index)] = [start, end]

        return ts_series_out

    def set_param_value(self, param_name: str, param_df: pd.DataFrame):
        """set the parameter value for the given parameter to the provided value.

        Parameters
        ----------
        param_name : str
            The name of the parameter

        param_df: pd.DataFrame
            The DataFrame value for this parameter.
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
        """get all the parameter DataFrame values for all of the parameters"""

        return self._param_df_list

    def get_param_df_for_param(self, param_name: str) -> pd.DataFrame:
        """get the parameter DataFrame values for the given parameter

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
        """get the parameter DataFrame values for the given parameter

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
        Returns the default time window duration and the default timeseries DataFrame
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

        return (
            self._param_file_exists,
            float(self._min_time_duration_before),
            float(self._min_time_duration_after),
        )

    def set_min_time_duration_values(
        self, min_time_duration_before: float, min_time_duration_after: float
    ):
        """set the parameter DataFrame values for the given parameter and also
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
            raise ValueError(
                "Min time duration file",
                param_min_t_file,
                "cannot be created or written to.",
            )
            pass

    def get_ts_series_for_timestamps(
        self, param_name: str, ts_start: float, ts_end: float
    ) -> list:
        """get the timestamp for the given parameter name. This routine just gets the
        timestamp series for a given parameter and calls `get_ts_split_for_ts_series`.
        See the doc for the other routine for details.
        """
        ts_split = []
        # For empty parameter, everything is considered to be in range.
        if param_name == "":
            ts_split.append([ts_start, ts_end, True])
            return ts_split

        ts_series = self.get_param_values_as_series(param_name)
        return self.get_ts_split_for_ts_series(ts_series, ts_start, ts_end)

    def get_ts_series_for_combined_param(self, ts_start: float, ts_end: float) -> list:
        """get the timestamp for all the parameters combined. This routine will combine
        the timestamp for all the parameters and then call `get_ts_split_for_ts_series`.
        See the doc for the other routine for details.
        """

        ts_series = self.get_combined_params_ts_series()
        return self.get_ts_split_for_ts_series(ts_series, ts_start, ts_end)

    def get_ts_split_for_ts_series(
        self, ts_series: list, ts_start: float, ts_end: float
    ) -> list:
        """get the timestamp split for the given ts series.

        Parameters
        ----------
        ts : list
            The timestamp list

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
        Timestamps: [[10, 15], [20, 25], [30, 35], [40, 45]]
        So, for a given timestamp of [8, 23] (i.e from 8 to 17 seconds), this routine will return:
        [[10, 15, True], [15, 20, False], [20, 23, True]]

        Another example:
        Window duration: 5s
        Timestamps: 10, 20
        So, for a given timestamp of [5, 35] (i.e from 5 to 35 seconds), this routine will return:
        [[5, 10, False], [10, 15, True], [15, 20, False], [20, 25, True], [25, 35, False]]
        """
        ts_split = []
        start_col_name = Parameters.PARAM_TIME_WINDOW_START_LIST
        end_col_name = Parameters.PARAM_TIME_WINDOW_END_LIST
        indices = list(
            filter(
                lambda x: (
                    ts_series.iloc[x][start_col_name] >= ts_start
                    and ts_series.iloc[x][start_col_name] <= ts_end
                )
                or (
                    (ts_series.iloc[x][start_col_name] < ts_start)
                    and ts_series.iloc[x][end_col_name] > ts_start
                ),
                range(len(ts_series)),
            )
        )

        # No timestamp in the series fits within the provided time. Mark
        # the whole duration as outside.
        if len(indices) == 0:
            ts_split.append([ts_start, ts_end, False])
            return ts_split

        start = ts_start
        idx = 0
        delta = 0.0
        while True:
            if start < ts_series.iloc[indices[idx]][start_col_name]:
                is_in = False
                end = min(
                    ts_end,
                    ts_series.iloc[indices[idx]][start_col_name]
                    - Parameters.TIME_PRECISION,
                )
            else:
                is_in = True
                end = min(ts_end, ts_series.iloc[indices[idx]][end_col_name])
                idx += 1

            ts_split.append([start + delta, end, is_in])
            if is_in:
                delta = Parameters.TIME_PRECISION
                if end == ts_end:
                    break
            else:
                end += Parameters.TIME_PRECISION
                delta = 0.0
                if end > ts_end:
                    break

            # If we have reached the end of the ts series and there is
            # still some left in the duration, just add the rest.
            if idx >= len(indices):
                if end < ts_end:
                    ts_split.append([end + delta, ts_end, False])
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
        """Parses the DataFrame into the window duration and the time series.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to parse.

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
                            "Min time duration",
                            line,
                            "from file",
                            param_min_t_file,
                            "cannot be converted to a number. Using default of",
                            t_duration,
                        )
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
            raise ValueError(
                "Parameter folder",
                self._param_dir,
                "does not exist!",
            )

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
