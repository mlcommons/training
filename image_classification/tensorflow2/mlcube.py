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
        env.update(
            {"DATA_DIR": data_dir,}
        )

        process = subprocess.Popen("./download_dataset.sh", cwd=".", env=env)
        process.wait()


@app.command("download")
def download(data_dir: str = typer.Option(..., "--data_dir")):
    DownloadTask.run(data_dir)


@app.command("train")
def train(data_dir: str = typer.Option(..., "--data_dir")):
    pass


if __name__ == "__main__":
    app()
