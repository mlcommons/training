import os

from constants import *


class SubmissionReport(object):
  """SubmissionReport stores submission info and produces submission report."""

  def __init__(self):
    self.passed_checks = []
    self.failed_checks = []
    self.errors = []
    self.results = {}

  def add_passed_check(self, msg):
    self.passed_checks.append(msg)

  def add_failed_check(self, msg):
    self.failed_checks.append(msg)

  def add_error(self, msg):
    self.errors.append(msg)

  def set_results(self, results):
    self.results = results

  def print_report(self, verbose=True):

    # print summary
    print("\n" + "MLPERF SUBMISSION REPORT\n" + "========================\n")
    print("\n" + "SUMMARY\n" + "-------\n")
    print("Passed checks: {}\n".format(len(self.passed_checks)) +
          "Failed checks: {}\n".format(len(self.failed_checks)) +
          "Errors: {}".format(len(self.errors)))
    print("Note: the Errors indicate certain checks being skipped or " +
          "early terminated due to errors. The errors are likely caused " +
          "by failed checks above.")

    # print succeeded checks (in verbose mode only)
    if verbose:
      print("\n" + "PASSED CHECKS\n" + "-------------\n")
      for msg in self.passed_checks:
        print(msg)

    # print failed checks
    print("\n" + "FAILED CHECKS\n" + "-------------\n")
    for msg in self.failed_checks:
      print(msg)

    # print errors
    print("\n" + "ERRORS\n" + "------\n")
    for msg in self.errors:
      print(msg)

  def print_results(self):
    result_columns = RESULT_SUBM_META_COLUMNS + RESULT_ENTRY_META_COLUMNS + BENCHMARK_NAMES

    print("\n" + "RESULTS\n" + "-------\n")
    print(",".join(result_columns))
    for entry_name in self.results:
      value_list = [self.results[entry_name][key] for key in result_columns]
      value_list = ["-" if val is None else str(val) for val in value_list]
      print(",".join(value_list))
