import logging
import warnings

from cpath import get_bert_config_path
from trainer_v2.chair_logging import IgnoreFilter
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config
from trainer_v2.custom_loop.per_task.trainer import Trainer
from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.custom_loop.train_loop import tf_run


def adjust_logging():
    msgs = [
        # "UserWarning: `layer.apply` is deprecated and will be removed in a future version",
        "`model.compile_metrics` will be empty until you train or evaluate the model."
    ]
    tf_logging = logging.getLogger("tensorflow")
    tf_logging.addFilter(IgnoreFilter(msgs))
    warnings.filterwarnings("ignore", '`layer.updates` will be removed in a future version. ')
    # warnings.filterwarnings("ignore", "`layer.apply` is deprecated")


def tf_run_for_bert(dataset_factory, model_config,
                    run_config: RunConfig2, inner):
    adjust_logging()
    run_config.print_info()

    bert_params = load_bert_config(get_bert_config_path())
    trainer = Trainer(bert_params, model_config, run_config, inner)
    tf_run(run_config, trainer, dataset_factory)