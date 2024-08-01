# U-Net3D MLPerf v1.0 Pytorch implementation.

## Features

The following features are supported by this model.
 
| **Feature** | **3D-UNet** |
|---------------------------------|-----|
| Distributed training            | Yes |
| Automatic mixed precision (AMP) | Yes |
| Learning rate decay             | Yes |
| Learning rate warm up           | Yes |
| Gradient accumulation           | Yes |
| Checkpointing                   | Yes |
| MLPerf logger                   | Yes |

**Distributed training**

This implementation takes advantage of the [Distributed communication package](https://pytorch.org/docs/stable/distributed.html) for multi-worker training.

**Automatic Mixed Precision (AMP)**
 
Passing `--amp` flag will enable Mixed Precision training and evaluation
 
**Learning rate decay**
 
`MultiStepLR` scheduler can be enabled by passing flag `--lr_decay_epochs` with one or milestones. For example, to have a change of learning rate at epochs 1000 and 2000 you can use
```bash
--learning_rate 1.0 --lr_decay_epochs 1000 2000 --lr_decay_factor 0.1
```
This will keep the learning rate at `1.0` between epochs 0 and 1000. Decrease it to `0.1` between epochs 1000 and 2000, and further decrease it to `0.01` at epoch 2000.

**Learning rate warm up**
 
A linear learning rate warm up schedule can be invoked by adding
```bash
--init_learning_rate 1e-4 --learning_rate 1.0 --lr_warmup_epochs 200
```
This will increase the learning rate from `1e-4` to `1.0` in the first 200 epochs of the training.

**Checkpointing**

To save a checkpoint during training you need to pass a path where to save the checkpoint, for example
```bash
--save_ckpt_path /results/checkpoint.pth
```
The script will save two checkpoints at that location - one with the latest state and with the best state. 
The best state is chosen based on `mean_dice` metric. For details please have a look at `runtime.callbacks.CheckpointCallback`.

To load a checkpoint for evaluation you have to invoke
```bash
--exec_mode evaluate --load_ckpt_path /results/checkpoint.pth
```

this will load the best checkpoint from `/results/checkpoint.pth` (or any given path).

**MLPerf logger**

MLPerf logger was added in compliance with the [official repository](https://github.com/mlcommons/logging).


## Quick Start Guide
  
1. Download the data
   
    The data is stored in the [KiTS19 github repository](https://github.com/neheller/kits19). To download it please follow the instructions:
    ```bash
    mkdir raw-data-dir
    cd raw-data-dir
    git clone https://github.com/neheller/kits19
    cd kits19
    pip3 install -r requirements.txt
    python3 -m starter_code.get_imaging
    ```
    This will download the original, non-interpolated data to `raw-data-dir/kits19/data`

2. Clone the repository.
 
    Create a folder for the project anc clone the repository
    
    ```bash
    git clone https://github.com/mmarcinkiewicz/training/tree/Add_unet3d
    cd training/image_segmentation/pytorch
    ```

3. Build the U-Net Docker container.
    
    ```bash
    docker build -t unet3d .
    ```
 
4. Start an interactive session in the container to run preprocessing/training/inference.
 
    You will need to mount two (or three) directories:
    - for raw data (RAW-DATA-DIR) 
    - for preprocessed data (PREPROCESSED-DATA-DIR)
    - (optionally) for results (RESULTS-DIR)
    
    ```bash
    mkdir data
    mkdir results
    docker run --ipc=host -it --rm --runtime=nvidia -v RAW-DATA-DIR:/raw_data -v PREPROCESSED-DATA-DIR:/data -v RESULTS-DIR:/results unet3d:latest /bin/bash
    ```
 
5. Preprocess the dataset.
    
    The data preprocessing script is called `preprocess_dataset.py`. All the required hyperparameters are already set. All you need to do is to invoke the script with correct paths:
    ```bash
    python3 preprocess_dataset.py --data_dir /raw_data --results_dir /data
    ```
   
    The script will preprocess each volume and save it as a numpy array at `/data`. It will also display some statistics like the volume shape, mean and stddev of the voxel intensity.

6. Start training.
  
    The basic command to run on 1 worker takes form:
    ```bash
    python3 main.py --data_dir /data/ --loader pytorch --log_dir /results/ --epochs 4000 --seed 0 --batch_size 2 --learning_rate 1.0 --eval_every 20
    ```
   
    Running this command for seeds in range `{0, 1, ..., 9}` should converge to the target accuracy `mean_dice` = 0.91.

## Repository content
 
In the root directory, the most important files are:
* `main.py`: Serves as the entry point to the application. Encapsulates the training routine.
* `Dockerfile`: Container with the basic set of dependencies to run U-Net3D.
* `requirements.txt`: Set of extra requirements for running U-Net3D.
* `preprocess_data.py`: Converts the dataset to numpy format for training.
* `evaluation_cases.txt`: A list of cases used for evaluation - a fixed split of the whole dataset.
 
The `data_loading/` folder contains the necessary load data. Its main components are:
* `data_loader.py`: Implements the data loading.
* `pytorch_loader.py`: Implements the data augmentation and iterators.
 
The `model/` folder contains information about the building blocks of U-Net3D and the way they are assembled. Its contents are:
* `layers.py`: Defines the different blocks that are used to assemble U-Net3D.
* `losses.py`: Defines the different losses used during training and evaluation.
* `unet3d.py`: Defines the model architecture using the blocks from the `layers.py` file.

The `runtime/` folder contains scripts with training and inference logic. Its contents are:
* `arguments.py`: Implements the command-line arguments parsing.
* `callbacks.py`: Collection of performance, evaluation, and checkpoint callbacks.
* `distributed_utils.py`: Defines a set of functions used for distributed training.
* `inference.py`: Defines the evaluation loop and sliding window.
* `logging.py`: Defines the MLPerf logger.
* `training.py`: Defines the training loop.
 
## Parameters
 
The complete list of the available parameters for the main.py script contains:

### Input/Output parameters
* `--data_dir`: Set the input directory containing the dataset (Required, default: `None`).
* `--log_dir`: Set the output directory for logs (default: `/tmp`).
* `--save_ckpt_path`: Path with a filename to save the checkpoint to (default: `None`). 
* `--load_ckpt_path`: Path with a filename to load the checkpoint from (default: `None`). 
* `--loader`: Loader to use (default: `pytorch`).
* `--local_rank`: Local rank for distributed training (default: `os.environ.get("LOCAL_RANK", 0)`).

### Runtime parameters
* `--exec_mode`: Select the execution mode to run the model (default: `train`). Modes available:
  * `train` - trains a model with given parameters. 
  * `evaluate` - loads checkpoint (if available) and performs evaluation on validation subset.
* `--batch_size`: Size of each minibatch per GPU (default: `2`).
* `--ga_steps`: Number of steps for gradient accumulation (default: `1`).
* `--epochs`: Maximum number of epochs for training (default: `1`).
* `--evaluate_every`: Epoch interval for evaluation (default: `20`).
* `--start_eval_at`: First epoch to start running evaluation at (default: `1000`).
* `--layout`: Data layout (default: `NCDHW`. `NDHWC` is not implemented).
* `--input_shape`: Input shape for images during training (default: `[128, 128, 128]`).
* `--val_input_shape`: Input shape for images during evaluation (default: `[128, 128, 128]`).
* `--seed`: Set random seed for reproducibility (default: `-1` - picks a random number from `/dev/urandom`).
* `--num_workers`: Number of workers used for dataloading (default: `8`).
* `--benchmark`: Enable performance benchmarking (disabled by default). If the flag is set, the script runs in a benchmark mode - each iteration is timed and the performance result (in images per second) is printed at the end.
* `--warmup_steps`: Used only for during benchmarking - the number of steps to skip (default: `200`). First iterations are usually much slower since the graph is being constructed. Skipping the initial iterations is required for a fair performance assessment.
* `--amp`: Enable automatic mixed precision (disabled by default).

### Optimizer parameters
* `--optimizer`: Type of optimizer to use (default: `sgd`, choices=`sgd, adam, lamb`).
* `--learning_rate`: Learning rate (default: `1.0`).
* `--momentum`: Momentum for SGD optimizer (default: `0.9`).
* `--init_learning_rate`: Initial learning rate used for learning rate warm up (default: `1e-4`).
* `--lr_warmup_epochs`: Number of epochs for learning rate warm up (default: `0`).
* `--lr_decay_epochs`: Milestones for MultiStepLR learning rate decay (default: `None`).
* `--lr_decay_factor`: Factor for MultiStepLR learning rate decay (default: `1.0`).
* `--lamb_betas`: Beta1 and Beta2 parameters for LAMB optimizer (default: `0.9, 0.999`).
* `--weight_decay`: Weight decay factor (default: `0.0`).

### Other parameters
* `--verbose`: Whether to display `tqdm` progress bars during training (default: `False`).
* `--oversampling`: Oversampling for biased crop (default: `0.4`).
* `--overlap`: Overlap for sliding window (default: `0.5`).
* `--cudnn_benchmark`: Whether to use cuDNN benchmark (default: `False`).
* `--cudnn_deterministic`: Whether to use cuDNN deterministic (default: `False`).
