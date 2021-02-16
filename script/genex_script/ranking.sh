export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=3
export LOGGER=4
model_path=/mnt/scratch/youngwookim/Chair/output/model/runs/BERT_Base_trained_on_MSMARCO/model.ckpt-100000

python src/explain/genex/runner/ranking_score.py  \
    --model_path=$model_path \
    --data_name=clue
