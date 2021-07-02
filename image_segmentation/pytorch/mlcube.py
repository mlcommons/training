"""MLCube handler file"""
import os
import yaml
import typer
import shutil
import subprocess
from pathlib import Path


app = typer.Typer()

class DownloadDataTask(object):
    """Download task Class
    It defines the environment variables:
        DATA_ROOT_DIR: Directory path to download the dataset
    Then executes the download script"""
    @staticmethod
    def run(data_dir: str) -> None:

        env = os.environ.copy()
        env.update({
            'DATA_ROOT_DIR': data_dir,
        })

        process = subprocess.Popen("./download_dataset.sh", cwd=".", env=env)
        process.wait()

class PreprocessDataTask(object):
    """Preprocess dataset task Class
    It defines the environment variables:
        DATA_DIR: Dataset directory path
        PROCESSED_DIR: Processed dataset path
    Then executes the preprocess script"""
    @staticmethod
    def run(data_dir: str, processed_dir: str) -> None:

        """
        args = f"--data_dir {data_dir} --results_dir {processed_dir}"
        command = "python preprocess_dataset.py " + args
        process = subprocess.call(command, cwd=".", shell=True)
        """

        env = os.environ.copy()
        env.update({
            'DATA_DIR': data_dir,
            'PROCESSED_DIR': processed_dir,
        })

        process = subprocess.Popen("./preprocess_dataset.sh", cwd=".", env=env)
        process.wait()

class TrainTask(object):
    """Preprocess dataset task Class
    It defines the environment variables:
        DATASET_DIR: Dataset directory path
        RESULT_DIR: Directory path where logs will be saved
        All other parameters defined in parameters_file
    Then executes the benchmark script"""
    @staticmethod
    def run(input_dir: str, output_dir: str, parameters_file: str) -> None:
        with open(parameters_file, 'r') as stream:
            parameters = yaml.safe_load(stream)

        env = os.environ.copy()
        env.update({
            'DATASET_DIR': input_dir,
            'RESULT_DIR': output_dir,
        })

        env.update(parameters)

        process = subprocess.Popen("./run_and_time.sh", cwd=".", env=env)
        process.wait()

@app.command("download_data")
def download_data(data_dir: str = typer.Option(..., '--data_dir')):
    DownloadDataTask.run(data_dir)

@app.command("preprocess_data")
def preprocess_data(data_dir: str = typer.Option(..., '--data_dir'),
           processed_dir: str = typer.Option(..., '--processed_dir')):
    PreprocessDataTask.run(data_dir, processed_dir)

@app.command("train")
def train(input_dir: str = typer.Option(..., '--input_dir'),
          output_dir: str = typer.Option(..., '--output_dir'),
          parameters_file: str = typer.Option(..., '--parameters_file')):
    TrainTask.run(input_dir, output_dir, parameters_file)

if __name__ == '__main__':
    app()
