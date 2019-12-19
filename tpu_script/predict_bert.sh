#!/usr/bin/env bash
export PYTHONPATH=src
python3 src/tlm/training/dynamic_mask_main.py \
	--bert_config_file=data/config/bert_config.json \
	--input_file=gs://clovertpu/training/pair_test/801 \
	--init_checkpoint=gs://clovertpu/training/bert_model/bert_model.ckpt \
	--output_dir=gs://clovertpu/training/model/unmasked_8 \
	--max_seq_length=512 \
	--save_checkpoints_steps=200 \
	--out_file=bert_801.pickle \
    --iterations_per_loop=100 \
	--train_batch_size=256 \
	--learning_rate=1e-5 \
	--num_train_steps=10000 \
	--do_predict \
	--use_tpu=True \
	--max_predictions_per_seq=20 \
	--tpu_name=v3-tf2-6

