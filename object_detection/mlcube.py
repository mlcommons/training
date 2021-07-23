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

class TrainTask(object):
    """Preprocess dataset task Class
    It defines the environment variables:
        DATA_DIR: Dataset directory path
        OUTPUT_DIR: Directory path where model will be saved
        All other parameters defined in parameters_file
    Then executes the benchmark script"""
    @staticmethod
    def run(data_dir: str, output_dir: str, parameters_file: str) -> None:
        with open(parameters_file, 'r') as stream:
            parameters = yaml.safe_load(stream)

        env = os.environ.copy()
        env.update({
            'DATA_DIR': data_dir,
            'OUTPUT_DIR': output_dir,
        })

        env.update(parameters)

        process = subprocess.Popen("./run_and_time.sh", cwd=".", env=env)
        process.wait()

@app.command("download_data")
def download_data(data_dir: str = typer.Option(..., '--data_dir')):
    DownloadDataTask.run(data_dir)

@app.command("train")
def train(data_dir: str = typer.Option(..., '--data_dir'),
          output_dir: str = typer.Option(..., '--output_dir'),
          parameters_file: str = typer.Option(..., '--parameters_file')):
    TrainTask.run(data_dir, output_dir, parameters_file)

if __name__ == '__main__':
    app()