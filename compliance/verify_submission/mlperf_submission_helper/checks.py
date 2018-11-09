"""Checks to run against a submission."""
from __future__ import print_function

import json
import os

import constants
import mlp_compliance.mlp_compliance as mlp_compliance
import report as subm_report

INFINITE_TIME = 9999999999.99


class SubmissionChecks(object):
  """Submission checks."""

  def __init__(self):
    self.report = subm_report.SubmissionReport()
    self.submission_meta = {}
    self.result_meta = {}
    self.result_entry_meta = {}

  def verify_dirs_and_files(self, root_dir):
    self.verify_root_dir(root_dir)
    self.verify_code_dir(root_dir)
    self.verify_results_dir(root_dir)

  # verify_metadata must be called after verify_dirs_and_files()
  def verify_metadata(self):
    self.verify_submission_metadata()
    self.verify_result_entry_metadata()

  def exists(self, path, is_dir=False):
    exists_fn = os.path.isdir if is_dir else os.path.isfile
    if exists_fn(path):
      self.report.add_passed_check('Path exists: {}'.format(path))
    else:
      self.report.add_failed_check('Path not found: {}'.format(path))

  def name_in(self, path, ref_list):
    basename = os.path.basename(path)
    if basename in ref_list:
      self.report.add_passed_check('{} name is in {}.'.format(path, ref_list))
    else:
      self.report.add_failed_check('{} name not in {}.'.format(path, ref_list))

  def keys_match(self, keys, ref_keys, context=''):
    different_keys = set(keys).difference(set(ref_keys))
    if different_keys:
      self.report.add_failed_check('Keys in {} do not match expected: '.format(
          context) + 'unmatched keys: {}'.format(list(different_keys)))
    else:
      self.report.add_passed_check('Keys in {} match expected.'.format(context))

  def verify_root_dir(self, root_dir):
    result_dir = os.path.join(root_dir, 'results')
    code_dir = os.path.join(root_dir, 'code')
    submission_meta_file = os.path.join(root_dir, 'submission.json')

    self.exists(result_dir, is_dir=True)
    self.exists(code_dir, is_dir=True)
    self.exists(submission_meta_file, is_dir=False)

    try:
      with open(submission_meta_file) as f:
        self.submission_meta = json.load(f)
    except Exception as e:
      self.report.add_error('Unable to parse submission meatadata: {}'.format(
          str(e)))

  def verify_code_dir(self, root_dir):
    code_root_dir = os.path.join(root_dir, 'code')
    try:
      for code_name in os.listdir(code_root_dir):
        code_dir = os.path.join(code_root_dir, code_name)
        if not os.path.isdir(code_dir):
          continue
        self.name_in(code_dir, constants.BENCHMARK_NAMES + ['shared'])
        if code_name in constants.BENCHMARK_NAMES:
          self.exists(os.path.join(code_dir, 'README.md'))
          self.exists(os.path.join(code_dir, 'preproc_dataset.sh'))
    except Exception as e:
      self.report.add_error('Unable to verify code dir: {}'.format(str(e)))

  def verify_results_dir(self, root_dir):
    code_root_dir = os.path.join(root_dir, 'code')
    result_root_dir = os.path.join(root_dir, 'results')
    try:
      for entry_name in os.listdir(result_root_dir):
        entry_dir = os.path.join(result_root_dir, entry_name)
        if not os.path.isdir(entry_dir):
          continue
        entry_meta_file = os.path.join(entry_dir, 'entry.json')
        try:
          with open(entry_meta_file) as f:
            self.result_entry_meta[entry_name] = json.load(f)
        except Exception as e:
          self.report.add_error(
              'Unable to parse result entry metadata: {}'.format(str(e)))
        self.exists(entry_meta_file)
        for result_name in os.listdir(entry_dir):
          result_dir = os.path.join(entry_dir, result_name)
          if not os.path.isdir(result_dir):
            continue
          self.name_in(result_dir, constants.BENCHMARK_NAMES)
          self.exists(
              os.path.join(code_root_dir, result_name,
                           'setup_' + entry_name + '.sh'))
          self.exists(
              os.path.join(code_root_dir, result_name,
                           'run_and_time_' + entry_name + '.sh'))
          result_num = constants.REQUIRED_RESULT_NUM.get(result_name, 0)
          for i in range(result_num):
            log_file_name = 'result_' + str(i) + '.txt'
            self.exists(os.path.join(result_dir, log_file_name))
            self.result_meta.setdefault(entry_name, {})
            self.result_meta[entry_name].setdefault(
                result_name, [None for j in range(result_num)])
            division = self.result_entry_meta[entry_name].get('division')
            log_path = os.path.join(result_dir, log_file_name)
            dt = self.verify_and_extract_time(log_path, division, result_name)
            self.result_meta[entry_name][result_name][i] = dt
    except Exception as e:
      self.report.add_error('Unable to verify results dir: {}'.format(str(e)))

  def verify_submission_metadata(self):
    subm_meta_keys = self.submission_meta.keys()
    different_keys = set(subm_meta_keys).difference(set(constants.SUBM_META_PROPS))
    if different_keys:
      self.report.add_failed_check(
          'Keys in submission metadata do not match expected: ' +
          'unmatched keys: {}'.format(list(different_keys)))
    else:
      self.report.add_passed_check(
          'Keys in submission metadata match expected.')

  def verify_result_entry_metadata(self):
    for entry_name in self.result_entry_meta:
      entry_meta = self.result_entry_meta[entry_name]
      entry_meta_keys = entry_meta.keys()
      self.keys_match(
          entry_meta_keys,
          constants.ENTRY_META_PROPS,
          context='entry {} metadata'.format(entry_name))
      try:
        for node_meta in entry_meta['nodes']:
          node_meta_keys = node_meta.keys()
          self.keys_match(
              node_meta_keys,
              constants.NODE_META_PROPS,
              context='entry {} node metadata'.format(entry_name))
      except Exception as e:
        self.report.add_error(
            'Unable to verify node metadata for entry {}: {}'.format(
                entry_name, str(e)))

  def compile_results(self):
    results = {}
    for entry_name in self.result_meta:
      results.setdefault(entry_name, {})
      for key in constants.RESULT_SUBM_META_COLUMNS:
        results[entry_name][key] = self.submission_meta[key]
      for key in constants.RESULT_ENTRY_META_COLUMNS:
        results[entry_name][key] = self.result_entry_meta[entry_name][key]
      for benchmark_name in constants.BENCHMARK_NAMES:
        benchmark_results = self.result_meta[entry_name].get(
            benchmark_name, None)
        if not benchmark_results:
          results[entry_name][benchmark_name] = None
          continue
        if not all(benchmark_results):
          self.report.add_error('Benchmark results contain None values. ' +
                                'entry: {}, benchmark name: {}'.format(
                                    entry_name, benchmark_name))
          results[entry_name][benchmark_name] = None
          continue
        # special treatment for the NCF results
        if benchmark_name == 'ncf':
          possible_results = benchmark_results
          benchmark_results = []
          for pr in possible_results:
            # Skips results that were failures to converge, which are set to
            # INFINITE_TIME
            if pr != INFINITE_TIME:
              benchmark_results.append(pr)
            # Stops after it finds 50 results, which may not be the first
            # 50 by date unless they are sorted earlier.
            if len(benchmark_results) == 50:
              break
          if len(benchmark_results) != 50:
            raise Exception('NCF does not have 50 good results')
          if len(possible_results) != 100:
            raise Exception('NCF does not have 100 good results:{}'.
                            format(len(possible_results)))

        benchmark_results = sorted(benchmark_results)
        del benchmark_results[0]
        del benchmark_results[-1]
        result_val = (
            float(sum(benchmark_results)) / len(benchmark_results) /
            constants.REFERENCE_RESULTS[benchmark_name])
        results[entry_name][benchmark_name] = result_val
    self.report.set_results(results)

  def get_compliance(self, filename):
    """Get the compliance level of the output file."""
    if filename.startswith('/'):
      output_file = filename
    else:
      output_file = os.path.join(self.workspace, filename)

    print('Running Compliance Check on {}'.format(output_file))
    print('#' * 80)
    status, dt, qual, target = mlp_compliance.l2_check_file(output_file)
    print('#' * 80)

    if status:
      level = '2'
    else:
      status, dt, qual, target = mlp_compliance.l1_check_file(output_file)
      print('#' * 80)
      if status:
        level = '1'
      else:
        level = '0'

    success = status and qual and target and qual >= target
    return level, dt, qual, success

  # use submodule mlp_compliance (https://github.com/bitfort/mlp_compliance)
  def verify_and_extract_time(self, log_file, division, result_name):
    expected_level = constants.DIVISION_COMPLIANCE_CHECK_LEVEL.get(division,
                                                                   None)
    print(result_name)
    if expected_level is None:
      raise Exception('Unknown division: {}'.format(division))
    level, dt, _, success = self.get_compliance(log_file)
    if int(level) != expected_level:
      raise Exception('Error Level {} does not match needed level {}:{}'.
            format(level, expected_level, log_file))

    # Sets failure to converge to "infinite time" per the rules
    if success and dt:
      return dt
    else:
      print('Result was not a success set to INFINITE_TIME({})'.
            format(INFINITE_TIME))
      return INFINITE_TIME
