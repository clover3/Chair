from transformers import AutoTokenizer
import tensorflow as tf

from trainer_v2.per_project.transparency.splade_regression.data_loaders.pairwise_eval import load_pairwise_mmp_data, \
    PairwiseEval, build_pairwise_eval_dataset
from trainer_v2.per_project.transparency.splade_regression.modeling.regression_modeling import get_transformer_sparse_encoder
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def main():
    model_config = {
        "model_type": "distilbert-base-uncased",
        "max_seq_length": 256
    }
    strategy = get_strategy(True, "v2-1")
    target_partition = list(range(1000, 1001))

    eval_pairwise_triplet = load_pairwise_mmp_data(target_partition)
    with strategy.scope():
        new_model = get_transformer_sparse_encoder(model_config, False)
        dataset: tf.data.Dataset = build_pairwise_eval_dataset(
            eval_pairwise_triplet, model_config["model_type"], 16, model_config["max_seq_length"])
        dataset = strategy.experimental_distribute_dataset(dataset)
        eval_obj = PairwiseEval(dataset, strategy, new_model)
        ret = eval_obj.do_eval()
        print(ret)


if __name__ == "__main__":
    main()