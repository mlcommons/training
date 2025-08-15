#!/bin/bash
#SBATCH -N 9
#SBATCH --gpus-per-node 1
#SBATCH -t 04:00:00
#SBATCH --mem=0

set -e

: "${CONT_IMAGE_URL:?CONT_IMAGE_URL not set}"
: "${TOKENIZER_PATH:?TOKENIZER_PATH not set}"
: "${MERGED_C4_PATH:?MERGED_C4_PATH not set}"
: "${PREPROCESSED_PATH:?PREPROCESSED_PATH not set}"

container_maps="${TOKENIZER_PATH}:/tokenizer,${MERGED_C4_PATH}:/dataset,${PREPROCESSED_PATH}:/outputs"

for index in {0..7}; do
    srun --nodes=1 --ntasks-per-node=1 \
    --container-image=$CONT_IMAGE_URL --container-mounts $container_maps --no-container-entrypoint \
    python3 /opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input "/dataset/c4-train.en_${index}.json.gz" \
    --output-prefix "/outputs/c4-train.en_${index}" \
    --tokenizer-library huggingface --tokenizer-type /tokenizer \
    --dataset-impl mmap --workers 128 &
done

srun --nodes=1 --ntasks-per-node=1 \
    --container-image=$CONT_IMAGE_URL --container-mounts $container_maps --no-container-entrypoint \
    --output preprocess_outputs/dataset_preprocess_validation.out \
    python3 /opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input "/dataset/c4-validation-91205-samples.en.json.gz" \
    --output-prefix "/outputs/c4-validation-91205-samples.en" \
    --tokenizer-library huggingface --tokenizer-type /tokenizer \
    --dataset-impl mmap --workers 128 & 
wait
