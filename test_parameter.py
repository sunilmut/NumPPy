#!/usr/bin/python

import glob
import logging
import os
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from numpy import NAN as NAN

import common
from parameter import Parameters

"""
------------------------------------------------------------
            Unit tests for the parameter module
------------------------------------------------------------
# Run these tests using `python -m unittest test_parameter`
# To run just one test case:
#       python -m unittest test_parameter.ParameterTest.<unit test name>
# Ex:    python -m unittest test_parameter.ParameterTest.test_param_bvt
------------------------------------------------------------
"""
param1 = {
    Parameters.PARAM_TIME_WINDOW_START_LIST: [100, 200, 300, 400, 500],
    Parameters.PARAM_TIME_WINDOW_DURATION: [30, np.nan, np.nan, np.nan, np.nan],
}

param2 = {
    Parameters.PARAM_TIME_WINDOW_START_LIST: [1.0, 2, 3, 4.0, 5],
    Parameters.PARAM_TIME_WINDOW_DURATION: [30, np.nan, np.nan, np.nan, np.nan],
}

min_time_duration_validation_set = [[5, 10], [1.0, 2], [20.0, 30.0], [0, 0]]


class ParameterTest(unittest.TestCase):
    TEST_DATA_DIR = "test_data"
    TEST_TRASH_DIR = "trash"

    def setUp(self):
        self.param = Parameters()
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.DEBUG)
            self.logger.addHandler(handler)

    def validate_df(self, df, expected_data):
        expected_df = pd.DataFrame(expected_data)
        self.assertEqual(expected_df.equals(df), True)

    def reset(self) -> str:
        input_dir = ParameterTest.get_trash_dir()
        param_dir = Parameters.get_param_dir(input_dir)
        shutil.rmtree(param_dir, ignore_errors=True)
        path = Path(param_dir)
        path.mkdir(parents=True, exist_ok=True)

        return input_dir

    def validate_min_t_duration(self, input_dir):
        expected_param = Parameters()
        expected_param._set_param_dir(input_dir)

        # Min time duration values should be set to default values without any
        # min time duration parameter present.
        (
            param_file_exists,
            min_time_duration_before,
            min_time_duration_after,
        ) = expected_param.get_min_time_duration_values()
        self.assertEqual(param_file_exists == False, True)
        self.assertEqual(
            min_time_duration_before == Parameters.MIN_TIME_DURATION_BEFORE_DEFAULT,
            True,
        )
        self.assertEqual(
            min_time_duration_after == Parameters.MIN_TIME_DURATION_AFTER_DEFAULT, True
        )

        # Set some min time duration values, parse and make sure they match
        for row in min_time_duration_validation_set:
            validate_param = Parameters()
            expected_param.set_min_time_duration_values(row[0], row[1])
            validate_param.parse(input_dir)
            (
                param_file_exists,
                min_time_duration_before,
                min_time_duration_after,
            ) = expected_param.get_min_time_duration_values()
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
            Parameters.PARAM_TIME_WINDOW_START_LIST: [10, 20, 30],
            Parameters.PARAM_TIME_WINDOW_DURATION: [5, np.nan, np.nan],
        }
        ts1 = [5, 8]
        expected_out1 = [[5, 8, False]]

        ts2 = [5, 37]
        expected_out2 = [
            [5, 9.999999, False],
            [10, 15, True],
            [15.000001, 19.999999, False],
            [20, 25, True],
            [25.000001, 29.999999, False],
            [30, 35, True],
            [35.000001, 37, False],
        ]

        ts3 = [5, 23]
        expected_out3 = [
            [5, 9.999999, False],
            [10, 15, True],
            [15.000001, 19.999999, False],
            [20, 23, True],
        ]

        ts4 = [5, 13]
        expected_out4 = [[5, 9.999999, False], [10, 13, True]]

        ts5 = [20, 25]
        expected_out5 = [[20, 25, True]]

        ts6 = [10, 15]
        expected_out6 = [[10, 15, True]]

        ts7 = [30, 35]
        expected_out7 = [[30, 35, True]]

        ts8 = [30, 43]
        expected_out8 = [[30, 35, True], [35.000001, 43, False]]

        ts9 = [45, 70]
        expected_out9 = [[45, 70, False]]

        ts10 = [25, 30]
        expected_out10 = [[25, 29.999999, False], [30, 30, True]]

        expected_ts = [
            [ts1, expected_out1],
            [ts2, expected_out2],
            [ts3, expected_out3],
            [ts4, expected_out4],
            [ts5, expected_out5],
            [ts6, expected_out6],
            [ts7, expected_out7],
            [ts8, expected_out8],
            [ts9, expected_out9],
            [ts10, expected_out10],
        ]
        param = Parameters()
        PARAM_NAME = "param"
        df = pd.DataFrame(param_val)
        self.logger.debug("testing parameter: %s", param_val)
        for val in expected_ts:
            self.logger.debug("timestamp duration: %s", val[0])
            input_dir = self.reset()
            param._set_param_dir(input_dir)
            param.set_param_value(PARAM_NAME, df)
            ts_split = param.get_ts_series_for_timestamps(
                PARAM_NAME, val[0][0], val[0][1], 0
            )
            self.logger.debug("Timestamp splits: %s", ts_split)
            self.logger.debug("Expected: %s", val[1])
            self.assertEqual(ts_split == val[1], True)

    def test_get_ts_series_for_timestamps_with_timeshift(self):
        param_val = {
            Parameters.PARAM_TIME_WINDOW_START_LIST: [10, 20, 30],
            Parameters.PARAM_TIME_WINDOW_DURATION: [5, np.nan, np.nan],
        }
        ts1 = [10, 15]
        expected_out_0 = [[10, 15, True]]  # Not shifted
        expected_out_2 = [[10.0, 11.999999, False],
                          [12.0, 15, True]]  # Shifted by 2
        expected_out_5 = [[10.0, 14.999999, False],
                          [15.0, 15, True]]  # Shifted by 5
        expected_out_10 = [[10, 15, False]]  # Not shifted
        expected_ts = [
            [0, expected_out_0],
            [2, expected_out_2],
            [5, expected_out_5],
            [10, expected_out_10],
        ]
        PARAM_NAME = "param"
        param = Parameters()
        df = pd.DataFrame(param_val)
        self.logger.debug("testing parameter: %s", param_val)
        for val in expected_ts:
            self.logger.debug("timeshift duration: %s", val[0])
            param.set_param_value(PARAM_NAME, df)
            ts_split = param.get_ts_series_for_timestamps(
                PARAM_NAME, ts1[0], ts1[1], val[0]
            )
            self.logger.debug("Timestamp splits: %s", ts_split)
            self.logger.debug("Expected: %s", val[1])
            self.assertEqual(ts_split == val[1], True)

    def test_get_combined_params_ts_series(self):
        param_val_1 = {
            Parameters.PARAM_TIME_WINDOW_START_LIST: [10, 20, 30, 40],
            Parameters.PARAM_TIME_WINDOW_DURATION: [5, np.nan, np.nan, np.nan],
        }
        PARAM_NAME_1 = "param_1"

        param_val_2 = {
            Parameters.PARAM_TIME_WINDOW_START_LIST: [17, 37],
            Parameters.PARAM_TIME_WINDOW_DURATION: [15, np.nan],
        }
        PARAM_NAME_2 = "param_2"

        param_val_3 = {
            Parameters.PARAM_TIME_WINDOW_START_LIST: [1, 15, 17, 32, 35, 36, 52, 55],
            Parameters.PARAM_TIME_WINDOW_DURATION: [
                1,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        }
        PARAM_NAME_3 = "param_3"

        param = Parameters()
        input_dir = self.reset()
        param._set_param_dir(input_dir)
        df = pd.DataFrame(param_val_1)
        param.set_param_value(PARAM_NAME_1, df)
        df = pd.DataFrame(param_val_2)
        param.set_param_value(PARAM_NAME_2, df)
        expected_df = pd.DataFrame(
            {
                Parameters.PARAM_TIME_WINDOW_START_LIST: [10.0, 17.0, 37.0],
                Parameters.PARAM_TIME_WINDOW_END_LIST: [15.0, 35.0, 52.0],
            }
        )
        combinded_df = param.get_combined_params_ts_series(0)
        self.assertTrue(combinded_df.equals(expected_df))
        df = pd.DataFrame(param_val_3)
        param.set_param_value(PARAM_NAME_3, df)
        expected_df = pd.DataFrame(
            {
                Parameters.PARAM_TIME_WINDOW_START_LIST: [1.0, 10.0, 17.0, 55.0],
                Parameters.PARAM_TIME_WINDOW_END_LIST: [2.0, 16.0, 53.0, 56.0],
            }
        )
        combinded_df = param.get_combined_params_ts_series(0)
        self.assertTrue(combinded_df.equals(expected_df))
        ts_split = param.get_ts_series_for_combined_param(0, 60, 0)
        expected_split = [
            [0, 0.999999, False],
            [1.0, 2.0, True],
            [2.000001, 9.999999, False],
            [10.0, 16.0, True],
            [16.000001, 16.999999, False],
            [17.0, 53.0, True],
            [53.000001, 54.999999, False],
            [55.0, 56.0, True],
            [56.000001, 60, False],
        ]
        self.assertEqual(ts_split == expected_split, True)

    def test_real_data(self):
        input_dir = os.path.join(os.getcwd(), "test_data", "parameter")
        param = Parameters()
        param.parse(input_dir)
        print(param.get_param_name_list())
        csv_path = glob.glob(os.path.join(input_dir, "*.csv"))
        self.assertEqual(len(csv_path), 1)
        timeshift_val, _v = common.get_timeshift_from_input_file(csv_path[0])
        self.assertGreater(timeshift_val, 0)
        self.logger.info("Using timeshift value of %f", timeshift_val)
        ts_split = param.get_ts_series_for_combined_param(
            1041.040000, 1057.180000, timeshift_val
        )
        expected_split = [
            [1041.04, 1046.91, True],
            [1046.910001, 1049.91, True],
            [1049.910001, 1057.18, False],
        ]
        self.assertEqual(ts_split == expected_split, True)

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
