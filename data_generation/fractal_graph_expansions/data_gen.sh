DATASET=${DATASET:-ml-20m}
USER_MUL=${USER_MUL:-16}
ITEM_MUL=${ITEM_MUL:-32}
DATA_DIR=${DATA_DIR:-/data/cache}

DATA_PATH=${DATA_DIR}/${DATASET}x${USER_MUL}x${ITEM_MUL}/

mkdir -p ${DATA_PATH}
python run_expansion.py --input_csv_file ${DATA_DIR}/${DATASET}/ratings.csv --num_row_multiplier ${USER_MUL} --num_col_multiplier ${ITEM_MUL} --output_prefix ${DATA_PATH}
