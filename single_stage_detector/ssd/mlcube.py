import os
import yaml
import typer
import shutil
import subprocess
from pathlib import Path


app = typer.Typer()


class DownloadDataTask(object):

    urls = {
        "train2017": "http://images.cocodataset.org/zips/train2017.zip",
        "val2017": "http://images.cocodataset.org/zips/val2017.zip",
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    }

    @staticmethod
    def download(cache_dir: Path, data_dir: Path, name: str) -> None:
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        dest_dir = data_dir / name
        if dest_dir.exists():
            print(f"Data ({name}) has already been downloaded: {dest_dir}")
            return

        archive_name = Path(DownloadDataTask.urls[name]).name
        cached_archive = cache_dir / archive_name
        if not cached_archive.exists():
            print(f"Data ({name}) is not in cache ({cached_archive}), downloading ...")
            os.system(f"cd {cache_dir}; curl -O {DownloadDataTask.urls[name]};")

        dest_archive = data_dir / archive_name
        if cache_dir != data_dir:
            print(f"Copying data from {cached_archive} to {dest_archive} ...")
            shutil.copyfile(cached_archive, dest_archive)

        print(f"Extracting archive ({archive_name}) ...")
        os.system(f"cd {data_dir}; unzip {archive_name};")

        if cache_dir != data_dir and dest_archive.exists():
            print(f"Removing data archive ({dest_archive}) ...")
            os.remove(dest_archive)

    @staticmethod
    def run(cache_dir: str, data_dir: str) -> None:
        for archive_name in DownloadDataTask.urls.keys():
            DownloadDataTask.download(Path(cache_dir), Path(data_dir), archive_name)


class DownloadModelTask(object):
    url = "https://download.pytorch.org/models/resnet34-333f7ec4.pth"

    @staticmethod
    def run(model_dir: str) -> None:
        os.makedirs(model_dir, exist_ok=True)
        os.system(f"cd {model_dir}; curl -O {DownloadModelTask.url};")


class TrainTask(object):
    @staticmethod
    def run(data_dir: str, pretrained_backbone: str, parameters_file: str) -> None:
        with open(parameters_file, 'r') as stream:
            parameters = yaml.safe_load(stream)

        env = os.environ.copy()
        env.update({
            'DATASET_DIR': data_dir,
            'ADDITIONAL_PARAMS': f"--pretrained-backbone {pretrained_backbone}",
            'NUMEPOCHS': str(parameters.get('num_epochs', 1))
        })

        process = subprocess.Popen("./run_and_time.sh", cwd=".", env=env)
        process.wait()


@app.command("download_data")
def download_data(cache_dir: str = typer.Option(..., '--cache_dir'),
                  data_dir: str = typer.Option(..., '--data_dir')):
    DownloadDataTask.run(cache_dir, data_dir)


@app.command("download_model")
def download_model(model_dir: str = typer.Option(..., '--model_dir')):
    DownloadModelTask.run(model_dir)


@app.command("train")
def train(data_dir: str = typer.Option(..., '--data_dir'),
          pretrained_backbone: str = typer.Option(..., '--pretrained_backbone'),
          parameters_file: str = typer.Option(..., '--parameters_file')):
    TrainTask.run(data_dir, pretrained_backbone, parameters_file)


if __name__ == '__main__':
    app()
