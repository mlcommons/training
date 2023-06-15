import argparse
import datetime
import glob
import os
import sys
import time

import numpy as np
import torch
import torchvision

try:
    import lightning.pytorch as pl
except:
    import pytorch_lightning as pl

from functools import partial

from omegaconf import OmegaConf
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader, Dataset

try:
    from lightning.pytorch import seed_everything
    from lightning.pytorch.callbacks import Callback
    from lightning.pytorch.callbacks.early_stopping import EarlyStopping
    from lightning.pytorch.trainer import Trainer
    from lightning.pytorch.utilities import rank_zero_info
    LIGHTNING_PACK_NAME = "lightning.pytorch."
except:
    from pytorch_lightning import seed_everything
    from pytorch_lightning.callbacks import Callback
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.trainer import Trainer
    from pytorch_lightning.utilities import rank_zero_info
    LIGHTNING_PACK_NAME = "pytorch_lightning."

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config

import mlperf_logging_utils
from mlperf_logging import mllog
import mlperf_logging.mllog.constants as mllog_constants

from typing import Any, Dict, Optional, Type
from lightning.pytorch.utilities.types import STEP_OUTPUT



class DataLoaderX(DataLoader):
# A custom data loader class that inherits from DataLoader
    def __iter__(self):
        # Overriding the __iter__ method of DataLoader to return a BackgroundGenerator
        # This is to enable data laoding in the background to improve training performance
        return BackgroundGenerator(super().__iter__())


def get_parser(**parser_kwargs):
    # A function to create an ArgumentParser object and add arguments to it

    def str2bool(v):
        # A helper function to parse boolean values from command line arguments
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
    # Create an ArgumentParser object with specifies kwargs
    parser = argparse.ArgumentParser(**parser_kwargs)

    # Add vairous command line arguments with their default balues and descriptions
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="train",
        choices=["train", "validate"],
        help="run mode, train or validation",
    )
    parser.add_argument(
        "-v",
        "--validation",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="validation",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project",
    )
    parser.add_argument(
        "-c",
        "--ckpt",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="load pretrained checkpoint from stable AI",
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "--fid-threshold",
        type=int,
        default=None,  # TODO(ahmadki): change after finzliaing RCPs
        help="halt training once this FID validation score or a smaller one is achieved",
    )
    parser.add_argument(
        "--clip-threshold",
        type=int,
        default=None,  # TODO(ahmadki): change after finzliaing RCPs
        help="halt training once this CLIP validation score or a higher one is achieved",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="/results",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )

    return parser

# A function that returns the non-default arguments between two objects
def nondefault_trainer_args(opt):
    # create an argument parsser
    parser = argparse.ArgumentParser()
    # add pytorch lightning trainer default arguments
    parser = Trainer.add_argparse_args(parser)
    # parse the empty arguments to obtain the default values
    args = parser.parse_args([])
    # return all non-default arguments
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))

# A dataset wrapper class to create a pytorch dataset from an arbitrary object
class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# A function to initialize worker processes
def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        # divide the dataset into equal parts for each worker
        split_size = dataset.num_records // worker_info.num_workers
        # set the sample IDs for the current worker
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        # set the seed for the current worker
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)

# Provide functionality for creating data loadedrs based on provided dataset configurations
class DataModuleFromConfig(pl.LightningDataModule):

    def __init__(self,
                 batch_size,
                 train=None,
                 validation=None,
                 test=None,
                 predict=None,
                 wrap=False,
                 num_workers=None,
                 shuffle_test_loader=False,
                 use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        # Set data module attributes
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        # If a dataset is passed, add it to the dataset configs and create a corresponding dataloader method
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        # Instantiate datasets
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        # Instantiate datasets from the dataset configs
        self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)

        # If wrap is true, create a WrappedDataset for each dataset
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        # Check if the train dataset is iterable
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        # Set the worker initialization function of the dataset isiterable or use_worker_init_fn is True
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        # Return a DataLoaderX object for the train dataset
        return DataLoaderX(self.datasets["train"],
                           batch_size=self.batch_size,
                           num_workers=self.num_workers,
                           shuffle=False if is_iterable_dataset else True,
                           worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        # Check if the validation dataset is iterable
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        # Return a DataLoaderX object for the validation dataset
        return DataLoaderX(self.datasets["validation"],
                           batch_size=self.batch_size,
                           num_workers=self.num_workers,
                           worker_init_fn=init_fn,
                           shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        # Check if the test dataset is iterable
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        # Set the worker initialization function if the dataset is iterable or use_worker_init_fn is True
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoaderX(self.datasets["test"],
                           batch_size=self.batch_size,
                           num_workers=self.num_workers,
                           worker_init_fn=init_fn,
                           shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoaderX(self.datasets["predict"],
                           batch_size=self.batch_size,
                           num_workers=self.num_workers,
                           worker_init_fn=init_fn)


class SetupCallback(Callback):
    # I nitialize the callback with the necessary parameters

    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    # Save a checkpoint if training is interrupted with keyboard interrupt
    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    # Create necessary directories and save configuration files before training starts
    # def on_pretrain_routine_start(self, trainer, pl_module):
    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            # ################################## TODO(ahmadki): debug
            # if not os.path.exists(self.cfgdir):
            #     import time
            #     time.sleep(5)
            #     os.makedirs(self.logdir, exist_ok=True)
            #     os.makedirs(self.ckptdir, exist_ok=True)
            #     os.makedirs(self.cfgdir, exist_ok=True)
            #     os.makedirs(self.imgsdir, exist_ok=True)

            #     os.makedirs(self.logdir, exist_ok=True)
            #     os.makedirs(self.ckptdir, exist_ok=True)
            #     os.makedirs(self.cfgdir, exist_ok=True)
            #     os.makedirs(self.imgsdir, exist_ok=True)
            # ################################## TODO(ahmadki): debug end
            OmegaConf.save(self.config, os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            # Save project config and lightning config as YAML files
            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        # Remove log directory if resuming training and directory already exists
        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass

    # def on_fit_end(self, trainer, pl_module):
    #     if trainer.global_rank == 0:
    #         ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
    #         rank_zero_info(f"Saving final checkpoint in {ckpt_path}.")
    #         trainer.save_checkpoint(ckpt_path)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py

    def on_train_start(self, trainer, pl_module):
        rank_zero_info("Training is starting")

    # the method is called at the end of each training epoch
    def on_train_end(self, trainer, pl_module):
        rank_zero_info("Training is ending")

    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device.index) / 2**20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.strategy.reduce(max_memory)
            epoch_time = trainer.strategy.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    # Setup mllogger
    mllogger = mllog.get_mllogger()
    mlperf_logging_utils.submission_info(mllogger=mllogger,
                                         submission_benchmark=mllog_constants.STABLE_DIFFUSION,
                                         submission_division=mllog_constants.CLOSED,
                                         submission_org="reference_implementation",
                                         submission_platform="DGX-A100",
                                         submission_poc_name="Ahmad Kiswani",
                                         submission_poc_email="akiswani@nvidia.com",
                                         submission_status=mllog_constants.ONPREM)

    mllogger.start(key=mllog_constants.INIT_START)

    # get the current time to create a new logging directory
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    status = mllog_constants.ABORTED

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    # Veirfy the arguments are both specified
    if opt.name and opt.resume:
        raise ValueError("-n/--name and -r/--resume cannot be specified both."
                         "If you want to resume training in a new log folder, "
                         "use -n/--name in combination with --resume_from_checkpoint")

    # Check if the "resume" option is specified, resume training from the checkpoint if it is true
    ckpt = None
    if opt.resume:
        rank_zero_info("Resuming from {}".format(opt.resume))
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            rank_zero_info("logdir: {}".format(logdir))
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        # Finds all ".yaml" configuration files in the log directory and adds them to the list of base configurations
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        # Gets the name of the current log directory by splitting the path and taking the last element.
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            rank_zero_info("Using base config {}".format(opt.base))
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

        # Sets the checkpoint path of the 'ckpt' option is specified
        if opt.ckpt:
            ckpt = opt.ckpt

    # Create the checkpoint and configuration directories within the log directory.
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    # Sets the seed for the random number generator to ensure reproducibility
    mllogger.event(key=mllog_constants.SEED, value=opt.seed)
    seed_everything(opt.seed)

    # Intinalize and save configuratioon using teh OmegaConf library.
    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())

        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)

        # Check whether the accelerator is gpu
        if not trainer_config["accelerator"] == "gpu":
            del trainer_config["accelerator"]
            cpu = True
        else:
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        use_fp16 = trainer_config.get("precision", 32) == 16
        if use_fp16:
            config.model["params"].update({"use_fp16": True})
        else:
            config.model["params"].update({"use_fp16": False})

        if ckpt is not None:
            # If a checkpoint path is specified in the ckpt variable, the code updates the "ckpt" key in the "params" dictionary of the config.model configuration with the value of ckpt
            config.model["params"].update({"ckpt": ckpt})
            rank_zero_info("Using ckpt_path = {}".format(config.model["params"]["ckpt"]))

        model = instantiate_from_config(config.model)
        # trainer and callbacks
        trainer_kwargs = dict()

        # config the logger
        # Default logger configs to  log training metrics during the training process.
        # These loggers are specified as targets in the dictionary, along with the configuration settings specific to each logger.
        default_logger_cfgs = {
            "wandb": {
                "target": LIGHTNING_PACK_NAME + "loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "tensorboard": {
                "target": LIGHTNING_PACK_NAME + "loggers.TensorBoardLogger",
                "params": {
                    "save_dir": logdir,
                    "name": "diff_tb",
                    "log_graph": True
                }
            }
        }

        # Set up the logger for TensorBoard
        default_logger_cfg = default_logger_cfgs["tensorboard"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = default_logger_cfg
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # config the strategy, defualt is ddp
        if "strategy" in trainer_config:
            strategy_cfg = trainer_config["strategy"]
            strategy_cfg["target"] = LIGHTNING_PACK_NAME + strategy_cfg["target"]
        else:
            strategy_cfg = {
                "target": LIGHTNING_PACK_NAME + "strategies.DDPStrategy",
                "params": {
                    "find_unused_parameters": False
                }
            }

        trainer_kwargs["strategy"] = instantiate_from_config(strategy_cfg)

        # Set up various callbacks, including logging, learning rate monitoring, and CUDA management
        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {                           # callback to set up the training
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,                 # resume training if applicable
                    "now": now,
                    "logdir": logdir,                     # directory to save the log file
                    "ckptdir": ckptdir,                   # directory to save the checkpoint file
                    "cfgdir": cfgdir,                     # directory to save the configuration file
                    "config": config,                     # configuration dictionary
                    "lightning_config": lightning_config, # LightningModule configuration
                }
            },
            "learning_rate_logger": {                     # callback to log learning rate
                "target": "lightning.pytorch.callbacks.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",           # logging frequency (either 'step' or 'epoch')
                }
            },
            "cuda_callback": {                            # callback to handle CUDA-related operations
                "target": "main.CUDACallback"
            },
            # TODO(ahmadki): AND relation
            "fid_early_stop_callback" : {
                "target": "lightning.pytorch.callbacks.early_stopping.EarlyStopping",
                "params": {
                    "monitor": "validation/fid",
                    "stopping_threshold": opt.fid_threshold,
                    "mode": "min",                        # Minimize FID
                    "check_finite": True
                }
            },
            "clip_early_stop_callback" : {
                "target": "lightning.pytorch.callbacks.early_stopping.EarlyStopping",
                "params": {
                    "monitor": "validation/clip",
                    "stopping_threshold": opt.clip_threshold,
                    "mode": "max",                        # Maximize CLIP
                    "check_finite": True
                }
            }
        }

        # If the LightningModule configuration has specified callbacks, use those
        # Otherwise, create an empty OmegaConf configuration object
        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()


        # Merge the default callbacks configuration with the specified callbacks configuration, and instantiate the callbacks
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
        trainer_kwargs["callbacks"].append(mlperf_logging_utils.MLPerfLoggingCallback(mllogger=mllogger))

        # Set up ModelCheckpoint callback to save best models
        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": LIGHTNING_PACK_NAME + "callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}-{step:09}",
                "verbose": True,
                "save_last": True,
            }
        }

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg = OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        trainer_kwargs["callbacks"].append(instantiate_from_config(modelckpt_cfg))

        # Create a Trainer object with the specified command-line arguments and keyword arguments, and set the log directory
        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        trainer.logdir = logdir

        # Create a data module based on the configuration file
        data = instantiate_from_config(config.data)
        # TODO(ahmadki): check if this is necessary
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()

        # Configure gradient accumulation
        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        mllogger.event(mllog_constants.GRADIENT_ACCUMULATION_STEPS, value=accumulate_grad_batches)

        # Configure number of GPUs
        if not cpu:
            ngpu = trainer_config["devices"] * trainer_config["num_nodes"]
        else:
            ngpu = 1

        # Configure batch size
        local_batch_size = config.data.params.train.params.batch_size
        global_batch_size = local_batch_size*ngpu
        mllogger.event(mllog_constants.GLOBAL_BATCH_SIZE, value=trainer.world_size * local_batch_size)

        # Configure learning rate based on the batch size, base learning rate and number of GPUs
        # If scale_lr is true, calculate the learning rate based on additional factors
        base_lr = config.model.base_learning_rate
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * local_batch_size * base_lr
            rank_zero_info(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (local_batch_size) * {:.2e} (base_lr)"
                .format(model.learning_rate, accumulate_grad_batches, ngpu, local_batch_size, base_lr))
        else:
            model.learning_rate = base_lr
            rank_zero_info("++++ NOT USING LR SCALING ++++")
            rank_zero_info(f"Setting learning rate to {model.learning_rate:.2e}")

        # Allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb
                pudb.set_trace()

        import signal
        # Assign melk to SIGUSR1 signal and divein to SIGUSR2 signal
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # TODO(ahmadki): when does pytorch lightning initialize/touches data ?
        mllogger.end(mllog_constants.INIT_STOP)

        # Run the training and validation
        if opt.mode=="train":
            try:
                # TODO(ahmadki): when does pytorch lightning initialize/touches data ?
                mllogger.end(mllog_constants.INIT_STOP)
                trainer.fit(model, data)
            except Exception:
                melk()
                raise
        elif opt.mode=="validate":
            trainer.validate(model, data)
        else:
            raise ValueError(f"Unknown mode {opt.mode}")

        # TODO(ahmadki): AND relation
        # Default is True in case thresholds are not defined
        fid_success = True
        clip_success = True
        if opt.fid_threshold is not None:
            fid_success =  "validation/fid" in trainer.callback_metrics and opt.fid_threshold >= trainer.callback_metrics["validation/fid"].item()
        
        if opt.clip_threshold is not None:
            clip_success = "validation/fid" in trainer.callback_metrics and opt.clip_threshold <= trainer.callback_metrics["validation/clip"].item()

        status = mllog_constants.SUCCESS if fid_success and clip_success else mllog_constants.ABORTED

    except Exception:
        # If there's an exception, debug it if opt.debug is true and the trainer's global rank is 0
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        #  Move the log directory to debug_runs if opt.debug is true and the trainer's global
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer.global_rank == 0:
            print(trainer.profiler.summary())

        mllogger.event(mllog_constants.STATUS, value=status)
