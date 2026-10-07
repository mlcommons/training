set -e

: "${C4_PATH:?C4_PATH not set}"
: "${MERGED_C4_PATH:?MERGED_C4_PATH not set}"
: "${N_VALIDATION_SAMPLES:=91205}"
# defaults the N_VALIDATION_SAMPLES to 91205
# C4 validation dataset: each sample on average tokenizes to 518 tokens
# thus, to reach 47,185,920 validation tokens, we need to use at least 91205 samples,
# which, after tokenization, will yield 47,186,855 tokens. 

# create softlinks to store each shard before merging
mkdir -p softlinks
for shard in {0..7}; do
  start=$((shard * 128))
  end=$((shard * 128 + 127))
  mkdir -p softlinks/en_$shard
  for ind in $(seq -f "%05g" $start $end); do
    src=${C4_PATH}/c4-train.${ind}-of-01024.json.gz
    if [ -f "$src" ]; then
      ln -s "$src" softlinks/en_${shard}/
    else
      echo "Warning: missing file $src — skipping" >&2
    fi
  done
done

mkdir -p softlinks/en_validation
start=0
end=7
for ind in $(seq -f "%05g" $start $end); do
  ln -s ${C4_PATH}/c4-validation.${ind}-of-00008.json.gz softlinks/en_validation/c4-validation.${ind}-of-00008.json.gz
done

# merge
for shard in {0..7}; do
  cat softlinks/en_${shard}/*gz > ${MERGED_C4_PATH}/c4-train.en_${shard}.json.gz 
done

cat softlinks/en_validation/*gz > ${MERGED_C4_PATH}/c4-validation.en.json.gz

# select the first N_VALIDATION_SAMPLES number of samples
zcat ${MERGED_C4_PATH}/c4-validation.en.json.gz | head -n $N_VALIDATION_SAMPLES | gzip > ${MERGED_C4_PATH}/c4-validation-${N_VALIDATION_SAMPLES}-samples.en.json.gz