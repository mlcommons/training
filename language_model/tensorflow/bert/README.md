To run this model, use the following command.

```shell

python run_pretraining.py \
  --bert_config_file=./bert_config.json \
  --output_dir=/tmp/output/ \
  --input_file="./uncased_seq_512/wikipedia.tfrecord*,./uncased_seq_512/books.tfrecord*" \
  --nodo_eval \
  --do_train \
  --eval_batch_size=8 \
  --init_checkpoint=./checkpoint/model.ckpt-7037 \
  --iterations_per_loop=1000 \
  --max_predictions_per_seq=76 \
  --max_seq_length=512 \
  --num_train_steps=1365333333 \
  --num_warmup_steps=0 \
  --optimizer=lamb \
  --save_checkpoints_steps=1000 \
  --start_warmup_step=0 \
  --train_batch_size=24/
