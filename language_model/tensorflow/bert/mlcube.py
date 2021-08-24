"""MLCube handler file"""
import os
import shutil
import subprocess
from pathlib import Path

import typer
import yaml

app = typer.Typer()


class DownloadTask(object):
    """Download task Class
    It defines the environment variables:
        DATA_ROOT_DIR: Directory path to download the dataset
    Then executes the download script"""
    @staticmethod
    def run(data_dir: str) -> None:

        env = os.environ.copy()
        env.update({
            'DATA_DIR': data_dir,
        })

        process = subprocess.Popen(
            "./cleanup_scripts/download_and_uncompress.sh", cwd=".", env=env)
        process.wait()


class ExtractTask(object):
    """Extract task Class
    It defines the environment variables:
        DATA_ROOT_DIR: Directory path to download the dataset
    Then executes the download script"""
    @staticmethod
    def run(data_dir: str) -> None:

        env = os.environ.copy()
        env.update({
            'DATA_DIR': data_dir,
        })

        process = subprocess.Popen(
            "./cleanup_scripts/run_wiki_extractor.sh", cwd=".", env=env)
        process.wait()


class PreprocessTask(object):
    """Preprocess task Class
    It defines the environment variables:
        DATA_ROOT_DIR: Directory path to download the dataset
    Then executes the download script"""
    @staticmethod
    def run(data_dir: str) -> None:

        env = os.environ.copy()
        env.update({
            'DATA_DIR': data_dir,
        })
        process = subprocess.Popen(
            "./process_wiki.sh", cwd="./cleanup_scripts", env=env)
        process.wait()


class GenerateTfrecordsTask(object):
    """Preprocess task Class
    It defines the environment variables:
        DATA_ROOT_DIR: Directory path to download the dataset
    Then executes the download script"""
    @staticmethod
    def run(data_dir: str) -> None:

        env = os.environ.copy()
        env.update({
            'DATA_DIR': data_dir,
        })
        process = subprocess.Popen(
            "./generate_tfrecords.sh", cwd="./cleanup_scripts", env=env)
        process.wait()


class TrainTask(object):
    """Preprocess dataset task Class
    It defines the environment variables:
        DATA_DIR: Dataset directory path
        All other parameters are defined in the parameters_file
    Then executes the benchmark script"""
    @staticmethod
    def run(dataset_file_path: str, parameters_file: str) -> None:
        with open(parameters_file, 'r') as stream:
            parameters = yaml.safe_load(stream)

        pass


@app.command("download")
def download(data_dir: str = typer.Option(..., '--data_dir')):
    DownloadTask.run(data_dir)


@app.command("extract")
def extract(data_dir: str = typer.Option(..., '--data_dir')):
    ExtractTask.run(data_dir)


@app.command("preprocess")
def preprocess(data_dir: str = typer.Option(..., '--data_dir')):
    PreprocessTask.run(data_dir)


@app.command("generate_tfrecords")
def generate_tfrecords(data_dir: str = typer.Option(..., '--data_dir')):
    GenerateTfrecordsTask.run(data_dir)


@app.command("train")
def train(dataset_file_path: str = typer.Option(..., '--dataset_file_path'),
          parameters_file: str = typer.Option(..., '--parameters_file')):
    TrainTask.run(dataset_file_path, parameters_file)


if __name__ == '__main__':
    app()
