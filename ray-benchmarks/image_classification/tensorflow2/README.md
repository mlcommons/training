This directory contains scripts for running benchmarks.

## Machine
* I used a p3.2xlarge instance for initial testing. You'll need an AMI with the nvidia drivers and docker.
* The model requires tf 2.4.0. tf 2.9.0 fails due to experimental API changes.

## Data
* The full imagenet dataset is present at `s3://anyscale-data/imagenet/train/`.
* I did initial testing on a subset of this dataset, accessible at `s3://balajis-tiny-imagenet` (`aws s3 sync s3://balajis-tiny-imagenet .`).
* If we're running this a lot, to make data management easier, we should set up FSx for Lustre. It allows for quick startup time of an EC2 instance, which will have high-throughput access to the mnist dataset.

### Offline preprocessing (balajis-tiny-imagenet)

To do offline preprocessing, MLPerf uses a script present [here: imagenet_to_gcs.py](https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py).
I had to modify the script to work with the balajis-tiny-imagenet dataset:
```
diff --git a/tools/datasets/imagenet_to_gcs.py b/tools/datasets/imagenet_to_gcs.py
index 816ca63..569cf24 100644
--- a/tools/datasets/imagenet_to_gcs.py
+++ b/tools/datasets/imagenet_to_gcs.py
@@ -335,7 +335,7 @@ def convert_to_tf_records(

   # Glob all the training files
   training_files = tf.gfile.Glob(
-      os.path.join(raw_data_dir, TRAINING_DIRECTORY, '*', '*.JPEG'))
+      os.path.join(raw_data_dir, TRAINING_DIRECTORY, '*', 'images', '*.JPEG'))

   # Get training file synset labels from the directory name
   training_synsets = [
```

You'll have to create the `synset_labels.txt` file:
```
ln -s wnids.txt synset_labels.txt
```

You can run the script using the following arguments:
```
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    /usr/bin/python3.9 \
    imagenet_to_gcs.py \
    --gcs_upload=false \
    --raw_data_dir=/home/ubuntu/dev/data/balajis-tiny-imagenet \
    --local_scratch_dir=/home/ubuntu/dev/data/balajis-tiny-imagenet/tfrecords
```

## Running training
Start the docker container. The pathnames probably will have to be adjusted for your machine.
```
./run_docker
```
Then, from within the docker container, you can start training:
```
./train_model_docker
```

### Modifications made to training script
See the bottom of `train_model_docker` for a list of modifications I made to the training parameters. The TL;DR is that I made modifications to show more logs, use only 1 GPU (not data-parallel with 8), and a smaller batch size to fit into GPU memory.

We'll have to tweak more settings, I think right now the dataloader is using a large amount of GPU memory for prefetching. We can get much higher throughput by tuning these for
16GB GPU memory (instead of 16 * 8 aggregate memory with data-parallelism).
