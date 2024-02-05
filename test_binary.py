#!/usr/bin/python

import logging
import unittest

import numpy as np
import pandas as pd

import binary as b
import common
from parameter import *

"""
------------------------------------------------------------
            Unit tests for the binary.py module
------------------------------------------------------------
# Run these tests using `python -m unittest test_binary`
------------------------------------------------------------
"""
input_data1 = {
    common.INPUT_COL0_TS: [1, 2, 3, 4, 5, 10, 20, 30, 100, 200, 300, 310],
    common.INPUT_COL1_MI: [1, 2, 3, 4, 5, 6, 7.1, 8.2, 9.3, 10, 11, 20],
    common.INPUT_COL2_FREEZE: [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
}

output_data1 = {
    b.OUTPUT_COL0_TS: [1.0, 3.0, 4.0, 10.0, 20.0, 300.0],
    b.OUTPUT_COL1_MI: [1, 3, 4, 6, 7.1, 11],
    b.OUTPUT_COL2_MI_AVG: [1.5, 1.5, 4.5, 4.5, 8.65, 8.65],
    b.OUTPUT_COL3_FREEZE_TP: [
        b.ONE_TO_ZERO,
        b.ZERO_TO_ONE,
        b.ONE_TO_ZERO,
        b.ZERO_TO_ONE,
        b.ONE_TO_ZERO,
        b.ZERO_TO_ONE,
    ],
}

output_data1_min_t_1_before = {
    b.OUTPUT_COL0_TS: [1.0, 3.0, 4.0, 10.0, 20.0],
    b.OUTPUT_COL1_MI: [1, 3, 4, 6, 7.1],
    b.OUTPUT_COL2_MI_AVG: [1.5, 1.5, 4.5, 4.5, 8.65],
    b.OUTPUT_COL3_FREEZE_TP: [
        b.ONE_TO_ZERO,
        b.ZERO_TO_ONE,
        b.ONE_TO_ZERO,
        b.ZERO_TO_ONE,
        b.ONE_TO_ZERO,
    ],
}

output_data1_min_t_4_after = {
    b.OUTPUT_COL0_TS: [4.0, 10.0, 20.0],
    b.OUTPUT_COL1_MI: [4, 6, 7.1],
    b.OUTPUT_COL2_MI_AVG: [4.5, 4.5, 8.65],
    b.OUTPUT_COL3_FREEZE_TP: [b.ONE_TO_ZERO, b.ZERO_TO_ONE, b.ONE_TO_ZERO],
}

output_data1_min_t_5_before = {
    b.OUTPUT_COL0_TS: [10.0, 20.0],
    b.OUTPUT_COL1_MI: [6, 7.1],
    b.OUTPUT_COL2_MI_AVG: [4.5, 8.65],
    b.OUTPUT_COL3_FREEZE_TP: [b.ZERO_TO_ONE, b.ONE_TO_ZERO],
}

test_p1 = {
    Parameters.PARAM_TIME_WINDOW_START_LIST: [1, 100.0, 300.0],
    Parameters.PARAM_TIME_WINDOW_DURATION: [10, np.nan, np.nan],
}

out_p1 = {
    b.OUTPUT_COL0_TS: [1.0, 3.0, 4.0, 10.0, 300.0],
    b.OUTPUT_COL1_MI: [1.0, 3.0, 4.0, 6.0, 11.0],
    b.OUTPUT_COL2_MI_AVG: [1.5, 1.5, 4.5, 4.5, 8.65],
    b.OUTPUT_COL3_FREEZE_TP: [
        b.ONE_TO_ZERO,
        b.ZERO_TO_ONE,
        b.ONE_TO_ZERO,
        b.ZERO_TO_ONE,
        b.ZERO_TO_ONE,
    ],
}

out_p1_nop = {
    b.OUTPUT_COL0_TS: [20.0],
    b.OUTPUT_COL1_MI: [7.1],
    b.OUTPUT_COL2_MI_AVG: [8.65],
    b.OUTPUT_COL3_FREEZE_TP: [b.ONE_TO_ZERO],
}


class TestBinary(common.CommonTetsMethods, unittest.TestCase):
    def setUp(self):
        self.input_df1 = pd.DataFrame(input_data1)
        self.p1_df = pd.DataFrame(test_p1)
        logging.basicConfig(filename=b.OUTPUT_LOG_FILE,
                            level=logging.DEBUG, format="")
        common.logger = logging.getLogger(__name__)
        if not common.logger.handlers:
            common.logger.addHandler(common.loghandler())

    def validate_df(self, df, expected_data):
        expected_df = pd.DataFrame(expected_data)
        self.assertTrue(expected_df.equals(df.astype(expected_df.dtypes)))

    def validate_min_t(self, min_t_before, min_t_after, df, exp_out):
        out_df = b.apply_min_time_duration_criteria(
            min_t_before, min_t_after, df)
        # print(out_df)
        out_df.reset_index(drop=True, inplace=True)
        self.validate_df(out_df, exp_out)

    def test_process_input_df(self):
        exp_out_df = pd.DataFrame(output_data1)
        result, out_df = b.process_input_df(self.input_df1)
        self.assertTrue(result)
        self.assertTrue(exp_out_df.equals(out_df.astype(exp_out_df.dtypes)))
        self.validate_min_t(1, 0, out_df, output_data1_min_t_1_before)
        self.validate_min_t(5, 0, out_df, output_data1_min_t_5_before)
        self.validate_min_t(0, 4, out_df, output_data1_min_t_4_after)

    def test_param_processing(self):
        parameter_obj = Parameters()
        param_name = "test_p1"
        parameter_obj.set_param_value(param_name, self.p1_df)
        out_df = b.process_input_df(self.input_df1)[1]
        nop_df = out_df[:]
        temp_out_df, nop_df = b.process_param(
            parameter_obj, param_name, out_df, nop_df)
        temp_out_df.reset_index(drop=True, inplace=True)
        self.validate_df(temp_out_df, out_p1)
        self.validate_df(nop_df, out_p1_nop)

    def test_real_data(self):
        input_dir = os.path.join(os.getcwd(), "test_data", "binary")
        output_dir_base = os.path.join(
            os.getcwd(), "test_data", "binary_output")
        expected_output_dir_base = os.path.join(
            os.getcwd(), "test_data", "binary_output_expected"
        )
        successfully_parsed_files, unsuccessfully_parsed_files = b.main(
            input_dir, False, None
        )
        self.assertEqual(len(unsuccessfully_parsed_files), 0)
        success_normalized_path = []
        for file in successfully_parsed_files:
            success_normalized_path.append(os.path.normpath(file))

        num_files_compared = 0
        csv_path = glob.glob(os.path.join(input_dir, "*.csv"))
        # Validate that each of the input files were successfully parsed and matches expected
        for input_csv_file in csv_path:
            self.assertTrue((input_csv_file in success_normalized_path))
            file_name_without_ext, ext = os.path.splitext(
                os.path.basename(input_csv_file)
            )
            expected_output_dir = os.path.join(
                expected_output_dir_base, file_name_without_ext
            )
            common.logger.info("Expected output dir: %s", expected_output_dir)
            output_dir = os.path.join(output_dir_base, file_name_without_ext)
            common.logger.info("Output dir: %s", output_dir)
            csv_path = glob.glob(os.path.join(expected_output_dir, "*.csv"))
            for expected_csv_file in csv_path:
                file_name = os.path.basename(expected_csv_file)
                output_csv_file = os.path.join(output_dir, file_name)
                super().compare_csv_files(expected_csv_file, output_csv_file)
                num_files_compared += 1

        # Make sure that at least 1 file was compared
        common.logger.info(
            "Number of files successfully compared: %d", num_files_compared
        )
        self.assertGreater(num_files_compared, 0)
