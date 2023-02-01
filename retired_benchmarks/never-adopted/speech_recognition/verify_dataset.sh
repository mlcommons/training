# Script to verify the dataset

#generate tar, this takes a few minutes
tar -cf data-LibriSpeech-ref.tar LibriSpeech_dataset

#generate checksum on tar, this takes a few minutes
cksum data-LibriSpeech-ref.tar > data-LibriSpeech-cksum.out

#check against ref checksum and report success/failure
cmp --silent data-LibriSpeech-cksum.out data/data-LibriSpeech-ref-cksum.out && echo 'Dataset Checksum Passed.' || echo 'WARNING: Dataset Checksum Failed.'

#remove generated checksum and tar
rm data-LibriSpeech-ref.tar
rm data-LibriSpeech-cksum.out
