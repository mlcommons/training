#!/bin/bash
set -e

DATA_PATH="./dataset"
MODEL_PATH="./models"

# Capture MLCube parameter
while [ $# -gt 0 ]; do
  case "$1" in
  --data_path=*)
    DATA_PATH="${1#*=}"
    ;;
  --model_path=*)
    MODEL_PATH="${1#*=}"
    ;;
  *) ;;
  esac
  shift
done

if ! command -v huggingface-cli &> /dev/null; then
    echo "Error: huggingface-cli is not installed. Please add 'huggingface-hub' to your pip requirements." >&2
    exit 1
fi


echo "--- Preparing Directories ---"
mkdir -p "$DATA_PATH"
cd "$DATA_PATH"
echo "Working directory: $(pwd)"


echo "--- Downloading and unzipping dataset ---"
curl -O https://storage.googleapis.com/mlperf_training_demo/flux/flux_minified_data.zip
unzip -o -q flux_minified_data.zip
rm flux_minified_data.zip
echo "Dataset downloaded successfully."


mkdir -p "$MODEL_PATH"
echo "--- Downloading models to ${MODEL_PATH} directory ---"
echo HUGGING_FACE_HUB_TOKEN $HUGGING_FACE_HUB_TOKEN

echo "Downloading FLUX.1-schnell autoencoder..."
huggingface-cli download black-forest-labs/FLUX.1-schnell ae.safetensors \
    --local-dir "${MODEL_PATH}/autoencoder" \
    --local-dir-use-symlinks False

echo "Downloading T5-v1_1-xxl text encoder..."
huggingface-cli download google/t5-v1_1-xxl \
    --local-dir "${MODEL_PATH}/t5" \
    --exclude "tf_model.h5" \
    --local-dir-use-symlinks False

echo "Downloading CLIP-vit-large-patch14 image encoder..."
huggingface-cli download openai/clip-vit-large-patch14 \
    --local-dir "${MODEL_PATH}/clip" \
    --exclude "*.safetensors,*.msgpack,tf_model.h5" \
    --local-dir-use-symlinks False

echo "--- All downloads completed successfully! ---"