import json
import os
import subprocess

from constants import *
import report as subm_report


class SubmissionChecks(object):

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
            self.report.add_passed_check("Path exists: {}".format(path))
        else:
            self.report.add_failed_check("Path not found: {}".format(path))

    def name_in(self, path, ref_list):
        basename = os.path.basename(path)
        if basename in ref_list:
            self.report.add_passed_check("{} name is in {}.".format(path, ref_list))
        else:
            self.report.add_failed_check("{} name not in {}.".format(path, ref_list))

    def keys_match(self, keys, ref_keys, context=""):
        different_keys = set(keys).difference(set(ref_keys))
        if different_keys:
            self.report.add_failed_check(
                    "Keys in {} do not match expected: ".format(context) +
                    "unmatched keys: {}".format(list(different_keys)))
        else:
            self.report.add_passed_check(
                    "Keys in {} match expected.".format(context))

    def verify_root_dir(self, root_dir):
        result_dir = os.path.join(root_dir, "results")
        code_dir = os.path.join(root_dir, "code")
        submission_meta_file = os.path.join(root_dir, "submission.json")

        self.exists(result_dir, is_dir=True)
        self.exists(code_dir, is_dir=True)
        self.exists(submission_meta_file, is_dir=False)

        try:
            with open(submission_meta_file) as f:
                self.submission_meta = json.load(f)
        except Exception as e:
            self.report.add_error(
                    "Unable to parse submission meatadata: {}".format(str(e)))

    def verify_code_dir(self, root_dir):
        code_root_dir = os.path.join(root_dir, "code")
        try:
            for code_name in os.listdir(code_root_dir):
                code_dir = os.path.join(code_root_dir, code_name)
                if not os.path.isdir(code_dir):
                    continue
                self.name_in(code_dir, BENCHMARK_NAMES + ["shared"])
                if code_name in BENCHMARK_NAMES:
                    self.exists(os.path.join(code_dir, "README.md"))
                    self.exists(os.path.join(code_dir, "preproc_dataset.sh"))
        except Exception as e:
            self.report.add_error(
                    "Unable to verify code dir: {}".format(str(e)))

    def verify_results_dir(self, root_dir):
        code_root_dir = os.path.join(root_dir, "code")
        result_root_dir = os.path.join(root_dir, "results")
        try:
            for entry_name in os.listdir(result_root_dir):
                entry_dir = os.path.join(result_root_dir, entry_name)
                if not os.path.isdir(entry_dir):
                    continue
                entry_meta_file = os.path.join(entry_dir, "entry.json")
                try:
                    with open(entry_meta_file) as f:
                        self.result_entry_meta[entry_name] = json.load(f)
                except Exception as e:
                    self.report.add_error(
                            "Unable to parse result entry metadata: {}".format(str(e)))
                self.exists(entry_meta_file)
                for result_name in os.listdir(entry_dir):
                    result_dir = os.path.join(entry_dir, result_name)
                    if not os.path.isdir(result_dir):
                        continue
                    self.name_in(result_dir, BENCHMARK_NAMES)
                    self.exists(os.path.join(code_root_dir, result_name,
                            "setup_" + entry_name + ".sh"))
                    self.exists(os.path.join(code_root_dir, result_name,
                            "run_and_time_" + entry_name + ".sh"))
                    result_num = REQUIRED_RESULT_NUM.get(result_name, 0)
                    for i in range(result_num):
                        log_file_name = "result_" + str(i) + ".txt"
                        self.exists(os.path.join(result_dir, log_file_name))
                        self.result_meta.setdefault(entry_name, {})
                        self.result_meta[entry_name].setdefault(
                                result_name, [None for j in range(result_num)])
                        division = self.result_entry_meta[entry_name].get("division")
                        self.result_meta[entry_name][result_name][i] = \
                                self.verify_and_extract_time(
                                os.path.join(result_dir, log_file_name), division)
        except Exception as e:
            self.report.add_error("Unable to verify results dir: {}".format(str(e)))

    def verify_submission_metadata(self):
        subm_meta_keys = self.submission_meta.keys()
        different_keys = set(subm_meta_keys).difference(set(SUBM_META_PROPS))
        if different_keys:
            self.report.add_failed_check(
                    "Keys in submission metadata do not match expected: " +
                    "unmatched keys: {}".format(list(different_keys)))
        else:
            self.report.add_passed_check(
                    "Keys in submission metadata match expected.")

    def verify_result_entry_metadata(self):
        for entry_name in self.result_entry_meta:
            entry_meta = self.result_entry_meta[entry_name]
            entry_meta_keys = entry_meta.keys()
            self.keys_match(entry_meta_keys, ENTRY_META_PROPS,
                    context="entry {} metadata".format(entry_name))
            try:
                for node_meta in entry_meta["nodes"]:
                    node_meta_keys = node_meta.keys()
                    self.keys_match(node_meta_keys, NODE_META_PROPS,
                            context="entry {} node metadata".format(entry_name))
            except Exception as e:
                self.report.add_error(
                        "Unable to verify node metadata for entry {}: {}".format(
                        entry_name, str(e)))

    def compile_results(self):
        results = {}
        for entry_name in self.result_meta:
            results.setdefault(entry_name, {})
            for key in RESULT_SUBM_META_COLUMNS:
                results[entry_name][key] = self.submission_meta[key]
            for key in RESULT_ENTRY_META_COLUMNS:
                results[entry_name][key] = self.result_entry_meta[entry_name][key]
            for benchmark_name in BENCHMARK_NAMES:
                benchmark_results = self.result_meta[entry_name].get(benchmark_name, None)
                if not benchmark_results:
                    results[entry_name][benchmark_name] = None
                    continue
                if not all(benchmark_results):
                    self.report.add_error("Benchmark results contain None values. " +
                            "entry: {}, benchmark name: {}".format(entry_name, benchmark_name))
                    results[entry_name][benchmark_name] = None
                    continue
                # special treatment for the NCF results
                if benchmark_name == "ncf":
                    possible_results = benchmark_results
                    benchmark_results = []
                    for pr in possible_results:
                        if pr is not None:
                            benchmark_results.append(pr)
                        if len(benchmark_results) == 50:
                            break
                benchmark_results = sorted(benchmark_results)
                del benchmark_results[0]
                del benchmark_results[-1]
                result_val = (float(sum(benchmark_results)) /
                        len(benchmark_results) / REFERENCE_RESULTS[benchmark_name])
                results[entry_name][benchmark_name] = result_val
        self.report.set_results(results)

    # use submodule mlp_compliance (https://github.com/bitfort/mlp_compliance)
    def verify_and_extract_time(self, log_file, division):
        level = DIVISION_COMPLIANCE_CHECK_LEVEL.get(division, None)
        if level is None:
            raise Exception("Unknown division: {}".format(division))
        mlp_compliance_script = os.path.join(
                os.path.dirname(__file__), "mlp_compliance/mlp_compliance.py")
        output_str = subprocess.check_output(
                ["python", mlp_compliance_script, "--level", str(level), log_file])
        output_str = output_str.decode('utf-8')
        success_flag = False
        result_time = None
        for line in output_str.split("\n"):
            if line.startswith("SUCCESS"):
                success_flag = True
            if line.startswith("Measured time:"):
                result_time = float(line.lstrip("Measured time:").strip())
        if success_flag and result_time is not None:
            return result_time
        else:
            raise Exception("Result verification failed: {}".format(log_file))
