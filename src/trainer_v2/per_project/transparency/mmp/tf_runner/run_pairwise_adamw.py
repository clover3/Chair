
import os

from trainer_v2.custom_loop.modeling_common.adam_decay import AdamWeightDecay
from trainer_v2.per_project.transparency.mmp.pairwise_modeling import get_model, ModelConfig

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformers import TFAutoModelForSequenceClassification

from trainer_v2.custom_loop.dataset_factories import get_classification_dataset, get_pairwise_dataset
from trainer_v2.custom_loop.definitions import ModelConfigType
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config
import sys
from trainer_v2.chair_logging import c_log, IgnoreFilter, IgnoreFilterRE
import tensorflow as tf

from taskman_client.wrapper3 import report_run3
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def main(args):
    c_log.info(__file__)
    run_config: RunConfig2 = get_run_config2(args)
    run_config.print_info()
    model_config = ModelConfig()

    def build_dataset(input_files, is_for_training):
        return get_pairwise_dataset(
            input_files, run_config, model_config, is_for_training)

    run_name = str(run_config.common_run_config.run_name)
    c_log.info("Run name: %s", run_name)

    strategy = get_strategy_from_config(run_config)
    with strategy.scope():
        train_dataset = build_dataset(run_config.dataset_config.train_files_path, True)
        if run_config.dataset_config.eval_files_path:
            eval_dataset = build_dataset(run_config.dataset_config.eval_files_path, False)
        else:
            eval_dataset = None
        c_log.info("Building model")
        # optimizer_factory = tf.keras.optimizers.experimental.AdamW
        optimizer_factory = AdamWeightDecay
        model = get_model(model_config, run_config, optimizer_factory)

        c_log.info("model.fit() train_step=%d", run_config.train_config.train_step)
        model.fit(train_dataset,
                  validation_data=eval_dataset,
                  epochs=1,
                  steps_per_epoch=run_config.train_config.train_step,
                  validation_steps=100,
                  )
        model.save(run_config.train_config.model_save_path)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


