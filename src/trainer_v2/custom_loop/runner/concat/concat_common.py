from cpath import get_bert_config_path
from trainer_v2.custom_loop.dataset_factories import get_classification_dataset
from trainer_v2.custom_loop.definitions import ModelConfigType
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config
from trainer_v2.custom_loop.neural_network_def.two_seg_concat import TwoSegConcat2
from trainer_v2.custom_loop.per_task.trainer import Trainer
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2_nli
from trainer_v2.custom_loop.train_loop import tf_run
from trainer_v2.custom_loop.trainer_if import TrainerIF


class ModelConfig(ModelConfigType):
    max_seq_length = 600
    num_classes = 3


def concat_common(args, combiner):
    run_config: RunConfig2 = get_run_config2_nli(args)
    run_config.print_info()
    bert_params = load_bert_config(get_bert_config_path())
    model_config = ModelConfig()
    task_model = TwoSegConcat2(combiner)
    trainer: TrainerIF = Trainer(bert_params, model_config, run_config, task_model)

    def build_dataset(input_files, is_for_training):
        return get_classification_dataset(input_files, run_config, model_config, is_for_training)

    ret = tf_run(run_config, trainer, build_dataset)

    return "{}\n{}".format(run_config, ret)
