# MLLog

## Install

Use one of the following ways to install.

- For development, you may download the latest version and install from local path:

```sh
git clone https://github.com/mlperf/training.git
pip install -e training/compliance
```

- Install from github at a specific commit (Replace COMMIT_HASH with actual commit hash, the double quotes are needed):
  ```
  pip install "git+https://github.com/mlperf/training@COMMIT_HASH#subdirectory=compliance"
  ```

Uninstall:

```sh
pip uninstall mlperf_compliance
```

## Use

[Here](examples/dummy_example.py) is an example of how to use the `mllog` package in benchmark codes. You can run the example by:

```sh
python3 -m mllog.examples.dummy_example 
```
