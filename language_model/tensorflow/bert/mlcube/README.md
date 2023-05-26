# MLCube for Bert

MLCube™ GitHub [repository](https://github.com/mlcommons/mlcube). MLCube™ [wiki](https://mlcommons.github.io/mlcube/).

## Project setup

An important requirement is that you must have Docker installed.

```bash
# Create Python environment and install MLCube Docker runner 
virtualenv -p python3 ./env && source ./env/bin/activate && pip install mlcube-docker
# Fetch the implementation from GitHub
git clone https://github.com/mlcommons/training && cd ./training/language_model/tensorflow/bert
```

Go to mlcube directory and study what tasks MLCube implements.

```shell
cd ./mlcube
mlcube describe
```

### MLCube tasks

Download dataset.

```shell
mlcube run --task=download_data -Pdocker.build_strategy=always
```

Process dataset.

```shell
mlcube run --task=process_data -Pdocker.build_strategy=always
```

Train SSD.

```shell
mlcube run --task=train -Pdocker.build_strategy=always
```

Run compliance checker.

```shell
mlcube run --task=check_logs -Pdocker.build_strategy=always
```

### Execute the complete pipeline

You can execute the complete pipeline with one single command.

```shell
mlcube run --task=download_data,process_data,train,check_logs -Pdocker.build_strategy=always
```

## TPU Training

For executing this benchmark using TPU you will need access to [Google Cloud Platform](https://cloud.google.com/), then you can create a project (Note: all the resources should be created in the same project) and after that, you will need to follow the next steps:

1. Create a TPU node

In the Google Cloud console, search for the Cloud TPU API page, then click Enable.

Then go to the virtual machine sections and select [TPUs](https://console.cloud.google.com/compute/tpus)

Select create TPU node, fill in all the needed parameters, the recommended TPU type in the [readme](../README.md#on-tpu-v3-128) is v3-128 and the recommended TPU software version is 2.4.0.

The 3 most important parameters you need to remember are: `project name`, `TPU name`, and `TPU Zone`.

After creating, click on the TPU name to see the TPU details, and copy the Service account (should int the format: <service-xxxxxxxxxxxx@cloud-tpu.iam.gserviceaccount.com>)

2. Create a Google Storage Bucket

Go to [Google Storage](https://console.cloud.google.com/storage/browser) and create a new Bucket, define the needed parameters.

In the bucket list select the checkbox for the bucket you just created, then click on permissions, after that click on add principal.

In the new principals field paste the Service account from step 1, and then for the roles select, Storage Legacy Bucket Owner, Storage Legacy Bucket Reader and Storage Legacy Bucket Writer. Then click on save, this will allow the TPU to save the checkpoints during training.

3. Create a VM instance

The idea is to create a virtual machine instance containing all the code we will execute using MLCube.

Go to [VM instances](https://console.cloud.google.com/compute/instances), then click on create instance and define all the needed parameters (No GPU needed).

**IMPORTANT:** In the section Identity and API access, check the option `Allow full access to all Cloud APIs`, this will allow the connection between this VM, the Cloud Storage Bucket and the TPU.

Start the VM, connect to it via SSH, then use this [tutorial](https://docs.docker.com/engine/install/debian/) to install Docker.

After installing Docker, clone the repo and install MLCube and follow the to install MLCube, then go to the path: `training/language_model/tensorflow/bert/mlcube`

There modify the file at `workspace/parameters.yaml` and replace it with your data for:

```yaml
output_gs: your_gs_bucket_name
tpu_name: your_tpu_instance_name
tpu_zone: your_tpu_zone
gcp_project: your_gcp_project
```

After that run the command:

```shell
mlcube run --task=train_tpu --mlcube=mlcube_tpu.yaml -Pdocker.build_strategy=always
```

This will start the MLCube task that internally in the host VM will send a gRPC with all the data to the TPU through gRPC, then the TPU will get the code to execute and the information of the Cloud Storage Bucket data and will execute the training workload.
