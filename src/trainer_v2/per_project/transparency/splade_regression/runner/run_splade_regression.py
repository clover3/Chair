import sys
from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2
from trainer_v2.custom_loop.train_loop import tf_run
from trainer_v2.custom_loop.trainer_if import TrainerIF
from trainer_v2.per_project.transparency.splade_regression.data_loaders.dataset_factories import \
    get_vector_regression_dataset
from trainer_v2.per_project.transparency.splade_regression.modeling.regression_modeling import get_regression_model
from trainer_v2.per_project.transparency.splade_regression.trainer_huggingface_init import TrainerHuggingfaceInit
from trainer_v2.train_util.arg_flags import flags_parser
from transformers import AutoTokenizer


@report_run3
def main(args):
    c_log.info(__file__)
    run_config: RunConfig2 = get_run_config2(args)
    run_config.print_info()

    model_config = {
        "model_type": "distilbert-base-uncased",
    }
    vocab_size = AutoTokenizer.from_pretrained(model_config["model_type"]).vocab_size

    def model_factory():
        new_model = get_regression_model(model_config)
        return new_model

    trainer: TrainerIF = TrainerHuggingfaceInit(
        model_config, run_config, model_factory)

    def build_dataset(input_files, is_for_training):
        return get_vector_regression_dataset(
            input_files, vocab_size, run_config, is_for_training)

    tf_run(run_config, trainer, build_dataset)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


