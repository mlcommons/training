# Script to train and time DeepSpeech 2 implementation

RANDOM_SEED=1
TARGET_ACC=24

python train.py --model_path models/deepspeech_7_layer_rnn_t$RANDOM_SEED.pth.tar --seed $RANDOM_SEED
