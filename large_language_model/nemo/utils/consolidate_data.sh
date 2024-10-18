set -e

: "${C4_PATH:?C4_PATH not set}"
: "${MERGED_C4_PATH:?MERGED_C4_PATH not set}"

# create softlinks to store each shard before merging
mkdir -p softlinks
for shard in {0..7}; do
  start=$((shard * 128))
  end=$((shard * 128 + 127))
  mkdir -p softlinks/en_$shard
  for ind in $(seq -f "%05g" $start $end); do
    ln -s ${C4_PATH}/c4-train.${ind}-of-01024.json.gz softlinks/en_${shard}/c4-train.${ind}-of-01024.json.gz
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