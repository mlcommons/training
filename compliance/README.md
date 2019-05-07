# MLPerf Compliance Logging Utilities and Helper Functions

## Install

For development, you may download the latest version and install from local path:

```sh
git clone https://github.com/mlperf/training.git
pip install training/compliance
```

For submission, you may want to use a versioned package for sake of reproducibility. Here are some possible ways to do it:

- Install from github using a given commit (Replace COMMIT_HASH below with actual commit hash, the double quotes are needed):
  ```
  pip install "git+https://github.com/mlperf/training@COMMIT_HASH#subdirectory=compliance"
  ```
- Or, install from pypi iff a known compatible version is available (The version used must be known as compatible with current mlperf version. Replace VERSION below with the actual version number):
  ```
  pip install mlperf_compliance==VERSION
  ```

Uninstall:

```sh
pip uninstall mlperf_compliance
```

## Use

By default, the logs are written to stdout. To log to a file, set environment variable `COMPLIANCE_FILE` in your run environment, e.g.:

```sh
export COMPLIANCE_FILE=/your/result/file.log
```

[Here](mlperf_compliance/examples/dummy_example.py) is an example of how to use the package in benchmark codes. You can run the example by:

```sh
python3 -m mlperf_compliance.examples.dummy_example 
```
