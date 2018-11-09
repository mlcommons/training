"""Tests checks module."""
from __future__ import print_function

import os
import unittest

import checks


class TestChecks(unittest.TestCase):

  def test_verify_and_extract_time_not_success(self):
    """Tests extract the cpu model name."""
    smi_test = 'unittest_files/10_mixed_results/result_7.txt'
    smi_test = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            smi_test)
    sub_check = checks.SubmissionChecks()
    dt = sub_check.verify_and_extract_time(smi_test, 'closed', 'ncf')
    self.assertEqual(dt, checks.INFINITE_TIME)

  def test_verify_and_extract_time_success(self):
    """Tests extract the cpu model name."""
    smi_test = 'unittest_files/10_mixed_results/result_2.txt'
    smi_test = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            smi_test)
    sub_check = checks.SubmissionChecks()
    dt = sub_check.verify_and_extract_time(smi_test, 'closed', 'ncf')
    self.assertEqual(dt, 210.3895456790924)


