"""
This requires the MLCube 2.0 configuration
"""
import os
import yaml
import click
import typing
from mlcube_docker.docker_run import DockerRun


def load_config(mlcube_config_path: str, user_config_path: str) -> typing.Dict:
    """Returns dictionary containing MLCube configuration"""
    # Load mlcube config data
    try:
        with open(mlcube_config_path) as stream:
            mlcube_config_data = yaml.load(stream.read(), Loader=yaml.SafeLoader)
    except IOError as exc:
        # If file doesn't exist throw the exception:
        # OSError: {PATH_TO}/mnist/mlcube.yaml: No such file or directory
        raise IOError("%s: %s" % (mlcube_config_path, exc.strerror))

    # Load user config data if file exists
    if os.path.isfile(user_config_path):
        with open(user_config_path) as stream:
            user_config_data = yaml.load(stream.read(), Loader=yaml.SafeLoader)
    else:
        return mlcube_config_data

    # Merge config data
    tmp = mlcube_config_data['container']
    mlcube_config_data['container'] = user_config_data['container']
    mlcube_config_data['container'].update(tmp)
    return mlcube_config_data


@click.group(name='mlcube')
def cli():
    pass


@cli.command(name='run', help='Run MLCube ML task.',
             context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option('--mlcube', required=False, type=str, help='Path to MLCube directory, default is current.')
@click.option('--platform', required=False, type=str, help='Platform to run MLCube, default is docker/podman.')
@click.option('--task', required=False, type=str, help='MLCube task name to run, default is `main`.')
@click.option('--workspace', required=False, type=str, help='Workspace path, default is `workspace` within '
                                                            'MLCube folder')
def run(mlcube: str, platform: str, task: str, workspace: str):
    mlcube_root = os.path.abspath(mlcube or os.getcwd())
    if os.path.isfile(mlcube_root):
        mlcube_root = os.path.dirname(mlcube_root)

    platform = platform or 'docker'
    if platform != 'docker':
        raise ValueError(f"Only `docker` platform is supported")

    task = task or 'main'
    workspace = workspace or os.path.join(mlcube_root, 'workspace')

    mlcube_config_data = load_config(
        os.path.join(str(mlcube_root), 'mlcube.yaml'),
        os.path.join(os.path.expanduser("~"), '.mlcube.yaml')
    )

    docker_runner = DockerRun(mlcube_config_data, root=mlcube_root, workspace=workspace, task=task)
    docker_runner.run()


if __name__ == "__main__":
    cli()