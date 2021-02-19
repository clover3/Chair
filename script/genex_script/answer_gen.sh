export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=3
export LOGGER=4

python src/explain/genex/runner/answer_gen.py \
    --data_name=tdlt \
    --method_name=replace_token

