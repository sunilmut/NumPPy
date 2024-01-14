import logging
import numpy as np
import common
from parameter import *
import os
import pandas as pd
import glob
import unittest
import ff

"""
------------------------------------------------------------
            Unit tests for the ff module
------------------------------------------------------------
# Run these tests using `python -m unittest test_ff`
------------------------------------------------------------
"""
class FfTest(unittest.TestCase):
    def setUp(self):
        progress = ff.loghandler()
        logging.basicConfig(filename=ff.OUTPUT_LOG_FILE,
                            level=logging.DEBUG, format='')
        common.logger = logging.getLogger(__name__)
        common.logger.addHandler(progress)

    def test_bvt(self):
        # Whole duration parameter
        param = Parameters()
        in_col_names = [common.INPUT_COL0_TS, common.INPUT_COL1_MI, common.INPUT_COL2_FREEZE]
        ts = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
              20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
        fz = [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,
              1,  1,  1,  0,  0,  0,  1,  0,  1,  0,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0]
        mi = [1] * 40
        list_of_tuples = list(zip(ts, mi, fz))
        binary_df = pd.DataFrame(list_of_tuples, columns=in_col_names)
        data = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
        timeshift_val = 0
        print(binary_df)
        success, results = ff.process(param, "", binary_df, timeshift_val, data, ts)
        self.assertTrue(success)
        auc_0s_sum = results[0]
        auc_0s_cnt = results[1]
        out_df_0s = results[2]
        auc_0s_sum_not = results[3]
        auc_0s_cnt_not = results[4]
        out_df_0s_not = results[5]
        auc_1s_sum = results[6]
        auc_1s_cnt = results[7]
        out_df_1s = results[8]
        auc_1s_sum_not = results[9]
        auc_1s_cnt_not = results[10]
        out_df_1s_not = results[11]
        self.assertEqual(auc_0s_cnt, fz.count(0))
        self.assertEqual(auc_0s_sum, 443)
        self.assertEqual(auc_0s_sum_not, 0)
        self.assertEqual(auc_0s_cnt_not, 0)
        self.assertTrue(out_df_0s_not.empty)
        self.assertEqual(auc_1s_sum, 337)
        self.assertEqual(auc_1s_cnt, fz.count(1))
        self.assertEqual(auc_1s_sum_not, 0)
        self.assertEqual(auc_1s_cnt_not, 0)
        self.assertTrue(out_df_1s_not.empty)
        #auc_0s_avg, sem_auc_0s_sum, sem_auc_0s_avg = compute_val(auc_0s_sum, auc_0s_cnt, out_df_0s)
        #auc_1s_avg, sem_auc_1s_sum, sem_auc_1s_avg = compute_val(auc_1s_sum, auc_1s_cnt, out_df_1s)
        #print(auc_0s_avg, sem_auc_0s_sum, sem_auc_0s_avg)
        #print(auc_1s_avg, sem_auc_1s_sum, sem_auc_1s_avg)
        o_data = {ff.OUTPUT_COL0_TS:       [0.0, 15.0, 23.0, 27.0, 29.0, 35.0],
                  ff.OUTPUT_COL1_LEN:      [9.0, 4.0,  2.0,  0.0,  0.0,  4.0],
                  ff.OUTPUT_COL2_MI_AVG:   [1.0, 1.0,  1.0,  1.0,  1.0,  1.0],
                  ff.OUTPUT_COL3_DATA_AUC: [45.0,85.0, 72.0, 27.0, 29.0, 185.0],
                  ff.OUTPUT_COL4_DATA_AVG: [4.5, 17.0, 24.0, 27.0, 29.0, 37.0]}
        out_df_0s_expected = pd.DataFrame(o_data)
        self.assertTrue(out_df_0s_expected.equals(out_df_0s))
        o_data = {ff.OUTPUT_COL0_TS:       [10.0, 20.0, 26.0, 28.0, 30.0],
                  ff.OUTPUT_COL1_LEN:      [4.0,  2.0,  0.0,  0.0,  4.0],
                  ff.OUTPUT_COL2_MI_AVG:   [1.0,  1.0,  1.0,  1.0,  1.0],
                  ff.OUTPUT_COL3_DATA_AUC: [60.0, 63.0, 26.0, 28.0, 160.0],
                  ff.OUTPUT_COL4_DATA_AVG: [12.0, 21.0, 26.0, 28.0, 32.0]}
        out_df_1s_expected = pd.DataFrame(o_data)
        self.assertTrue(out_df_1s_expected.equals(out_df_1s))

        # Parameter that encompasses the whole duration
        param_val = {
            Parameters.PARAM_TIME_WINDOW_START_LIST:    [0],
            Parameters.PARAM_TIME_WINDOW_DURATION:      [40]
        }
        param = Parameters()
        PARAM_NAME = "param1"
        df = pd.DataFrame(param_val)
        param.set_param_value(PARAM_NAME, df)
        success, results = ff.process(param, PARAM_NAME, binary_df, timeshift_val, data, ts)
        self.assertTrue(success)
        auc_0s_sum = results[0]
        auc_0s_cnt = results[1]
        out_df_0s = results[2]
        auc_0s_sum_not = results[3]
        auc_0s_cnt_not = results[4]
        out_df_0s_not = results[5]
        auc_1s_sum = results[6]
        auc_1s_cnt = results[7]
        out_df_1s = results[8]
        auc_1s_sum_not = results[9]
        auc_1s_cnt_not = results[10]
        out_df_1s_not = results[11]
        self.assertEqual(auc_0s_cnt, fz.count(0))
        self.assertEqual(auc_0s_sum, 443)
        self.assertEqual(auc_0s_sum_not, 0)
        self.assertEqual(auc_0s_cnt_not, 0)
        self.assertTrue(out_df_0s_not.empty)
        self.assertEqual(auc_1s_sum, 337)
        self.assertEqual(auc_1s_cnt, fz.count(1))
        self.assertEqual(auc_1s_sum_not, 0)
        self.assertEqual(auc_1s_cnt_not, 0)
        self.assertTrue(out_df_1s_not.empty)
        # TODO: test for comppute val
        #auc_0s_avg, sem_auc_0s_sum, sem_auc_0s_avg = compute_val(auc_0s_sum, auc_0s_cnt, out_df_0s)
        #auc_1s_avg, sem_auc_1s_sum, sem_auc_1s_avg = compute_val(auc_1s_sum, auc_1s_cnt, out_df_1s)
        #print(auc_0s_avg, sem_auc_0s_sum, sem_auc_0s_avg)
        #print(auc_1s_avg, sem_auc_1s_sum, sem_auc_1s_avg)
        self.assertTrue(out_df_0s_expected.equals(out_df_0s))
        self.assertTrue(out_df_1s_expected.equals(out_df_1s))

        # Restricted parameter
        param_val = {
            Parameters.PARAM_TIME_WINDOW_START_LIST:    [0, 10, 20, 30, 40],
            Parameters.PARAM_TIME_WINDOW_DURATION:      [5, np.nan, np.nan, np.nan, np.nan]
        }
        param = Parameters()
        PARAM_NAME = "param1"
        df = pd.DataFrame(param_val)
        param.set_param_value(PARAM_NAME, df)
        success, results = ff.process(param, PARAM_NAME, binary_df, timeshift_val, data, ts)
        self.assertTrue(success)
        auc_0s_sum = results[0]
        auc_0s_cnt = results[1]
        out_df_0s = results[2]
        auc_0s_sum_not = results[3]
        auc_0s_cnt_not = results[4]
        out_df_0s_not = results[5]
        auc_1s_sum = results[6]
        auc_1s_cnt = results[7]
        out_df_1s = results[8]
        auc_1s_sum_not = results[9]
        auc_1s_cnt_not = results[10]
        out_df_1s_not = results[11]
        self.assertEqual(auc_0s_cnt, 9)
        self.assertEqual(auc_0s_sum, 87)
        # The _Not's should be the rest
        self.assertEqual(auc_0s_cnt + auc_0s_cnt_not, fz.count(0))
        self.assertEqual(auc_0s_sum_not, 356)
        self.assertEqual(auc_1s_sum, 283)
        self.assertEqual(auc_1s_cnt, 13)
        self.assertEqual(auc_1s_cnt + auc_1s_cnt_not, fz.count(1))
        self.assertEqual(auc_1s_sum_not, 54)
        #auc_0s_avg, sem_auc_0s_sum, sem_auc_0s_avg = compute_val(auc_0s_sum, auc_0s_cnt, out_df_0s)
        #auc_1s_avg, sem_auc_1s_sum, sem_auc_1s_avg = compute_val(auc_1s_sum, auc_1s_cnt, out_df_1s)
        #print(auc_0s_avg, sem_auc_0s_sum, sem_auc_0s_avg)
        #print(auc_1s_avg, sem_auc_1s_sum, sem_auc_1s_avg)
        o_data = {ff.OUTPUT_COL0_TS:       [0.0,  23.0],
                  ff.OUTPUT_COL1_LEN:      [5.0,  2.0],
                  ff.OUTPUT_COL2_MI_AVG:   [1.0,  1.0],
                  ff.OUTPUT_COL3_DATA_AUC: [15.0, 72.0],
                  ff.OUTPUT_COL4_DATA_AVG: [2.5,  24.0]}
        out_df_0s_expected = pd.DataFrame(o_data)
        self.assertTrue(out_df_0s_expected.equals(out_df_0s))
        o_data = {ff.OUTPUT_COL0_TS:       [6.0,  15.0, 27.0, 29.0, 35.0],
                  ff.OUTPUT_COL1_LEN:      [3.0,  4.0,  0.0,  0.0,  4.0],
                  ff.OUTPUT_COL2_MI_AVG:   [1.0,  1.0,  1.0,  1.0,  1.0],
                  ff.OUTPUT_COL3_DATA_AUC: [30.0, 85.0, 27.0, 29.0, 185.0],
                  ff.OUTPUT_COL4_DATA_AVG: [7.5,  17.0, 27.0, 29.0, 37.0]}
        out_df_0s_not_expected = pd.DataFrame(o_data)
        self.assertTrue(out_df_0s_not_expected.equals(out_df_0s_not))
        o_data = {ff.OUTPUT_COL0_TS:       [10.0, 20.0, 30.0],
                  ff.OUTPUT_COL1_LEN:      [4.0,  2.0,  4.0],
                  ff.OUTPUT_COL2_MI_AVG:   [1.0,  1.0,  1.0],
                  ff.OUTPUT_COL3_DATA_AUC: [60.0, 63.0, 160.0],
                  ff.OUTPUT_COL4_DATA_AVG: [12.0, 21.0, 32.0]}
        out_df_1s_expected = pd.DataFrame(o_data)
        self.assertTrue(out_df_1s_expected.equals(out_df_1s))
        o_data = {ff.OUTPUT_COL0_TS:       [26.0, 28.0],
                  ff.OUTPUT_COL1_LEN:      [0.0,  0.0],
                  ff.OUTPUT_COL2_MI_AVG:   [1.0,  1.0],
                  ff.OUTPUT_COL3_DATA_AUC: [26.0, 28.0],
                  ff.OUTPUT_COL4_DATA_AVG: [26.0, 28.0]}
        out_df_1s_not_expected = pd.DataFrame(o_data)
        self.assertTrue(out_df_1s_not_expected.equals(out_df_1s_not))

    def test_real_data(self):
        input_dir = os.path.join(os.getcwd(), "test_data", "ff")
        parameter_obj = Parameters()
        try:
            parameter_obj.parse(input_dir)
        except ValueError as e:
            common.logger.warning(e)
        ff.main(input_dir, parameter_obj)
        expected_output_dir = os.path.join(os.getcwd(), "test_data", "ff_output_expected", "binary")
        output_dir = os.path.join(os.getcwd(), "test_data", "ff_output", "binary")
        csv_path = glob.glob(os.path.join(expected_output_dir, '*.csv'))
        # Validate that the files in the output dir match the expected output dir
        for expected_csv_file in csv_path:
            file_name = os.path.basename(expected_csv_file)
            output_csv_file = os.path.join(output_dir, file_name)
            common.logger.debug("\nComparing output file with expected.\n\tExpected: %s,\n\tOutput:%s", expected_csv_file, output_csv_file)
            with open(expected_csv_file, 'r') as t1, open(output_csv_file, 'r') as t2:
                expected_lines = t1.readlines()
                output_lines = t2.readlines()
                x = 0
                files_match = True
                for line in expected_lines:
                    if line != output_lines[x]:
                        common.logger.error("%s!=\n%s", line, output_lines[x])
                        files_match = False
                    x += 1
                if files_match:
                    common.logger.debug("Output matches expected.")
                else:
                    common.logger.debug("Output does not matches expected.")
