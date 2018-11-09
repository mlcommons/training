"""Tests checks module."""
from __future__ import print_function

import os
import unittest

import checks


class TestChecks(unittest.TestCase):

  def test_verify_and_extract_time_not_success(self):
    """Tests extract the cpu model name."""
    smi_test = 'unittest_files/10_mixed_results/result_7.txt'
    smi_test = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), smi_test)
    sub_check = checks.SubmissionChecks()
    dt, start_time = sub_check.verify_and_extract_time(smi_test, 'closed',
                                                       'ncf')
    self.assertEqual(dt, checks.INFINITE_TIME)
    self.assertEqual(start_time, 1541638706.6702664)

  def test_verify_and_extract_time_success(self):
    """Tests extract the cpu model name."""
    smi_test = 'unittest_files/10_mixed_results/result_2.txt'
    smi_test = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), smi_test)
    sub_check = checks.SubmissionChecks()
    dt, start_time = sub_check.verify_and_extract_time(smi_test, 'closed',
                                                       'ncf')
    self.assertEqual(dt, 210.3895456790924)
    self.assertEqual(start_time, 1541635651.95072)

  def test_add_result(self):
    """Tests adding result to metadata dict."""
    sub_check = checks.SubmissionChecks()
    meta = {}
    meta['entry_name'] = {}
    meta['entry_name']['result_name'] = {}
    sub_check._add_result(meta['entry_name']['result_name'],
                          1,
                          10.11,
                          343434343.91)
    result = meta['entry_name']['result_name'][1]
    self.assertEqual(result['dt'], 10.11)
    self.assertEqual(result['start_time'], 343434343.91)

  def test_sort_results(self):
    """tests sorting results."""
    results_dict = []
    results_dict.append(self._create_result_dict(23, 1.993))
    results_dict.append(self._create_result_dict(55, 19.993))
    results_dict.append(self._create_result_dict(1, 0.993))
    results_dict.append(self._create_result_dict(99, 999991.993))

    sub_check = checks.SubmissionChecks()
    sorted_results = sub_check._sorted_results(results_dict)
    self.assertEqual(sorted_results, [1, 23, 55, 99])

  def _create_result_dict(self, dt, start_time):
    result = {}
    result['dt'] = dt
    result['start_time'] = start_time
    return result
