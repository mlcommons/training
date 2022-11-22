# C++ Minigo

The current self-play and training pipeline is run using a C++ implementation
of Minigo.

## Set up

Minigo++ depends on the TensorFlow C++ libraries, but we have not yet set up
Bazel WORKSPACE and BUILD rules to automatically download and configure
TensorFlow so (for now at least) you must perform a manual step to build the
library.  This depends on `zip`, so be sure that package is installed first.

If you want to run on GPU, you will have to tell Bazel where to find your CUDA
install (note that recent versions of Bazel ignore `LD_LIBRARY_PATH`):

```shell
sudo sh -c "echo /usr/local/cuda/lib64 > /etc/ld.so.conf.d/cuda.conf"
sudo ldconfig
```

Now, you're ready to compile TensorFlow from source:

```shell
sudo apt-get install zip
./cc/configure_tensorflow.sh
```

If you want to compile for CPU and not GPU, then execute the following instead:

```shell
sudo apt-get install zip
TF_NEED_CUDA=0 ./cc/configure_tensorflow.sh
```

This will automatically perform the first steps of
[Installing TensorFlow from Sources](https://www.tensorflow.org/install/install_sources)
but instead of installing the TensorFlow package, it extracts the generated C++
headers into the `cc/tensorflow` subdirectory of the repo. The script then
builds the required TensorFlow shared libraries and copies them to the same
directory. The tensorflow cc\_library build target in `cc/BUILD` pulls these
header and library files together into a format the Bazel understands how to
link against.

Although Minigo uses TensorFlow as its inference engine by default, other
engines can be used (e.g. Cloud TPU, TensorRT, TensorFlow Lite). See the
Inferences engines section below for more details.

## Getting a model

The TensorFlow models can found on our public Google Cloud Storage bucket. A
good model to start with is 000990-cormorant from the v15 run. The C++ engine
requires the frozen model in a `.minigo` format. You can copy it locally as
follows:

```shell
mkdir -p saved_models
gsutil cp gs://minigo-pub/v15-19x19/models/000990-cormorant.minigo saved_models/
```

Minigo can also read models directly from Google Cloud Storage but it doesn't
currently perform any local caching, so you're better off copying the model
locally once instead of copying from GCS every time.

## Binaries

C++ Minigo is made up of several different binaries. All binaries can be run
with `--helpshort`, which will display the full list of command line arguments.


#### cc:simple\_example

A very simple example of how to perform self-play using the Minigo engine. Plays
a single game using a fixed number of readouts.

```shell
bazel build -c opt cc:simple_example
bazel-bin/cc/simple_example \
  --model=saved_models/000990-cormorant.minigo \
  --num_readouts=160
```

#### cc:selfplay

The self-play binary used in our training pipeline. Has a lot more functionality
that the simple example, including:

 - Play multiple games in parallel, batching their inferences together for
   better GPU/TPU utilization.
 - Automatic loading of the latest trained model.
 - Write SGF games & TensorFlow training examples to Cloud Storage or Cloud
   BigTable.
 - Flag file support with automatic reloading. This is used among other things
   to dynamically adjust the resign threshold to minimize the number of bad
   resigns.


```shell
bazel build -c opt cc:selfplay
bazel-bin/cc/selfplay \
  --model=saved_models/000990-cormorant.minigo \
  --num_readouts=160 \
  --parallel_games=1 \
  --output_dir=data/selfplay \
  --holdout_dir=data/holdout \
  --sgf_dir=sgf
```

#### cc:eval

Evaluates the performance of two models by playing them against each other over
multiple games in parallel.

```shell
bazel build -c opt cc:eval
bazel-bin/cc/eval \
  --eval_model=saved_models/000990-cormorant.minigo \
  --target_model=saved_models/000990-cormorant.minigo \
  --num_readouts=160 \
  --parallel_games=32 \
  --sgf_dir=sgf
```

#### cc:gtp

Play using the GTP protocol. This is also the binary we recommend using as a
backend for Minigui (see `minigui/README.md`).

```shell
bazel build -c opt cc:gtp
bazel-bin/cc/gtp \
  --model=saved_models/000990-cormorant.minigo \
  --num_readouts=160
```

## Running the unit tests

Minigo's C++ unit tests operate on both 9x9 and 19x19, and some tests are only
enabled for a particular board size. Consequently, you must run the tests
twice: once for 9x9 boards and once for 19x19 boards.

```shell
bazel test --define=board_size=9 cc/...  &&  bazel test cc/...
```

Note that Minigo is compiled for a 19x19 board by default, which explains the
lack of a `--define=board_size=19` in the second `bazel test` invocation.

## Running with Address Sanitizer

Bazel supports building with AddressSanitizer to check for C++ memory errors:

```shell
bazel build cc:selfplay \
  --copt=-fsanitize=address \
  --linkopt=-fsanitize=address \
  --copt=-fno-omit-frame-pointer \
  --copt=-O1
```

If you need to programatic control of the sanitizers at runtime, you will have
to make Bazel aware of the sanitizer include path. Edit `WORKSPACE`, changing
`path` as appropriate:

```
new_local_repository(
    name = "sanitizers",
    path = "/usr/lib/gcc/x86_64-linux-gnu/7/include/sanitizer/",
    build_file_content = """
package(default_visibility = ["//visibility:public"])
cc_library(
    name = "sanitizers",
    hdrs = glob(["**/*.h"])
)
"""
)
```

Then add `@sanitizers` as a dependency to your `BUILD` target and you'll be able
to include `asan_interface.h` and `lsan_interface.h`. This will allow you to do
things like suppress leak checking at exit by calling `__lsan_disable()`.


## Profiling

Minigo uses the C++ bindings for
[https://google.github.io/tracing-framework/](Web Tracing Framework) to profile
the code. To enable, compile with
`bazel build --copt=-DWTF_ENABLE cc:selfplay` followed by the rest of your build
arguments. Without `WTF_ENABLE` defined, all the profiling code should be
optimized away by the compiler.

By default, the `cc:selfplay` binary writes to the trace to
`/tmp/minigo.wtf-trace` but this path can be overridden using the `--wtf_trace`
flag.


## File format & Inference engines

C++ Minigo supports a variety of inference engines, input features, layouts and
data types. In order to determine what configuration to use, Minigo uses a
custom `.minigo` data format that wraps the raw model data with extra metadata.
Which engines are compiled into the C++ binaries are controlled by passing
Bazel `--define` arguments at compile time.

 - **tf**: peforms inference using the TensorFlow libraries built by
   `cc/configure_tensorflow.sh`. Compiled & used as the inference by default,
   disable the engine with `--define=tf=0`. The model should be a frozen
   TensorFlow `GraphDef` proto (as generated by `freeze_graph.py`).
 - **lite**: performs inference using TensorFlow Lite, which runs in software on
   the CPU.
   Compile by passing `--define=lite=1` to `bazel build`.
   The model should be a Toco-optimized TFLite flat buffer (see below).
 - **tpu**: perform inference on a Cloud TPU. Your code must run on a Cloud
   TPU-equipped VM for this to work.
   Compile by passing `--define=tpu=1` to `bazel build`.
   The model should be a frozen `GraphDef` proto that was generated by passing
   `--use_tpu=true` when running `freeze_graph.py`. The TPU address must also
   be specified with `--define=$TPU_ADDRESS`, when `$TPU_ADDRESS` is the TPU's
   gRPC address (e.g. `grpc://10.240.2.10:8470`).
 - **random**: a model that returns random samples from a normal distribution,
   which can be useful for bootstrapping the reinforcement learning pipeline.
   Use by passing
   `--model=random:$FEATURES:$LAYOUT:$SEED`, where `$FEATURES` is the type of
   model features (e.g.  `agz`, `mlperf07`), `$LAYOUT` is the feature tensor
   layout (either `nhwc` or `nchw`) and `$SEED` is a random seed (use `0` to
   choose one based on the operating system's entropy source).

## Compiling a TensorFlow Lite model

First, unwrap the `.minigo` model into a `.pb`, then run it through Toco, the
TensorFlow optimizing compiler:

```
python3 oneoffs/unwrap_model.py \
      --src_path model.minigo \
      --dst_path model.pb

BATCH_SIZE=8
./cc/tensorflow/bin/toco \
  --input_file=model.pb \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --output_file=model.tflite \
  --inference_type=FLOAT \
  --input_type=FLOAT \
  --input_arrays=pos_tensor \
  --output_arrays=policy_output,value_output \
  --input_shapes=8,19,19,17
```

Unwrapping the model will print its metadata. You will need to rewrap the model
with the same metadata but setting the engine to `tflite`:

```
python3 oneoffs/wrap_model.py \
  --src_path model.tflite \
  --dst_path model.minifo \
  --metadata engine=lite,input_features=agz,input_layout=nhwc,input_type=float,board_size=19
```

## Cloud TPU

Minigo supports running inference on Cloud TPU.
Build `//cc:concurrent_selfplay` with `--define=tpu=1` and run with
`--device=$TPU_NAME` (see above).

To freeze a model into a model that can be run on Cloud TPU, use
`freeze_graph.py`:

```
python freeze_graph.py \
  --model_path=$MODEL_PATH \
  --use_tpu=true \
  --tpu_name=$TPU_NAME \
  --num_tpu_cores=8
```

Where `$MODEL_PATH` is the path to your model (either a local file or one on
Google Cloud Storage), and `$TPU_NAME` is the gRPC name of your TPU, e.g.
`grpc://10.240.2.10:8470`. This can be found from the output of
`gcloud beta compute tpus list`.

This command **must** be run from a Cloud TPU-ready GCE VM.

This invocation to `freeze_graph.py` will replicate the model 8 times so that
it can run on all eight cores of a Cloud TPU. To take advantage of this
parallelism when running selfplay, `virtual_losses * parallel_games` must be at
least 8, ideally 128 or higher.

## 9x9 boards

The C++ Minigo implementation requires that the board size be defined at compile
time, using the `MINIGO_BOARD_SIZE` preprocessor define. This allows us to
significantly reduce the number of heap allocations performed. The build scripts
are configured to compile with `MINIGO_BOARD_SIZE=19` by default. To compile a
version that works with a 9x9 board, invoke Bazel with `--define=board_size=9`.

## Bigtable

Minigo supports writing eval and selfplay results to
[Bigtable](https://cloud.google.com/bigtable/).

Build with `--define=bt=1` and run with
`--output_bigtable=<PROJECT>,<INSTANCE>,<TABLE>`.

For eval this would look something like
```
bazel-bin/cc/eval \
  --model=<MODEL_1> \
  --model_two=<MODEL_2> \
  --parallel_games=4 \
  --num_readouts=32
  --sgf_dir=sgf \
  --output_bigtable=<PROJECT>,minigo-instance,games
```

See number of eval games with
```
cbt -project=<PROJECT> -instance=minigo-instance \
  read games columns="metadata:eval_game_counter"
```
See eval results with
```
cbt -project=<PROJECT> -instance=minigo-instance \
  read games prefix="e_"
```

## Style guide

The C++ code follows
[Google's C++ style guide](https://github.com/google/styleguide)
and we use cpplint to delint.

