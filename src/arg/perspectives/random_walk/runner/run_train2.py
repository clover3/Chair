

import tensorflow as tf

from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging
from tlm.model.aux_emb import BertWithAux
from tlm.model_cnfig import JsonConfig
from tlm.training.cls_model_fn_multiple_lr import model_fn_classification
from tlm.training.flags_wrapper import get_input_files_from_flags
from tlm.training.input_fn import input_fn_builder_aux_emb_classification
from tlm.training.train_config import TrainConfigEx
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    tf_logging.info("Train perspective classification with aux / use mutiple learning rate")
    config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)
    input_files = get_input_files_from_flags(FLAGS)
    input_fn = input_fn_builder_aux_emb_classification(input_files,
                                                       FLAGS,
                                                       is_training=FLAGS.do_train,
                                                       dim=100
                                                       )

    def group_b(name):
        return "aux_emb" in name

    model_fn = model_fn_classification(config, train_config, BertWithAux, 'feed_features', group_b)
    return run_estimator(model_fn, input_fn)


if __name__ == "__main__":
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()

