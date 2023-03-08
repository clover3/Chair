from transformers import AutoTokenizer
import tensorflow as tf

from trainer_v2.per_project.transparency.splade_regression.data_loaders.pairwise_eval import load_pairwise_eval_data, \
    PairwiseEval, build_pairwise_eval_dataset
from trainer_v2.per_project.transparency.splade_regression.modeling.regression_modeling import get_regression_model
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def main():
    model_config = {
        "model_type": "distilbert-base-uncased",
        "max_seq_length": 256
    }
    vocab_size = AutoTokenizer.from_pretrained(model_config["model_type"]).vocab_size
    strategy = get_strategy()

    eval_pairwise_triplet = load_pairwise_eval_data()
    with strategy.scope():
        new_model = get_regression_model(model_config)
        dataset: tf.data.Dataset = build_pairwise_eval_dataset(
            eval_pairwise_triplet, model_config["model_type"], 16, model_config["max_seq_length"])
        dataset = strategy.experimental_distribute_dataset(dataset)
        eval_obj = PairwiseEval(dataset, strategy, new_model)
        ret = eval_obj.do_eval()
        print(ret)


if __name__ == "__main__":
    main()