export PYTHONPATH=src
python3 src/tlm/training/runner/train_loss_predict.py \
	--bert_config_file=data/config/bert_config.json \
	--model_config_file=data/config/tlm_bfn_config_independent.json \
	--init_checkpoint=gs://clovertpu/training/bert_model/bert_model.ckpt \
	--input_file=gs://clovertpu/training/tlm_bfn_loss/0 \
	--output_dir=gs://clovertpu/training/model/tlm_bfn_loss \
	--save_checkpoints_steps=5000 \
	--is_bert_checkpoint=True \
	--max_seq_length=512 \
	--iterations_per_loop=1000 \
	--train_batch_size=128 \
	--learning_rate=1e-5 \
	--num_train_steps=70000 \
	--do_train \
	--use_tpu=True \
	--repeat_data=False \
	--tpu_name=v3-tf2-3
		
