#!/bin/bash
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --mem=0
#SBATCH --array=6-7%8
#SBATCH --requeue

C4_PATH=$1

srun --container-image nvcr.io/nvidia/pytorch:21.12-py3 \
 --container-mounts ${C4_PATH}:${C4_PATH} \
 bash -c \
 " git clone https://github.com/NVIDIA/NeMo.git; \
   cd NeMo && git checkout f3ad584b94170bc3ea197df29eb9ef9c96061730 && bash ./reinstall.sh; \
   python /workspace/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input ${C4_PATH}/en_merge/c4-train.en_\${SLURM_ARRAY_TASK_ID}.json.gz \
    --tokenizer-library sentencepiece \
    --tokenizer-model ${C4_PATH}/tokenizers/c4_spm/sentencepiece.model \
    --output-prefix ${C4_PATH}/preprocessed_c4_spm/c4_en_\${SLURM_ARRAY_TASK_ID}_c4_spm_text_document \
    --dataset-impl mmap \
    --workers 128 "
