import warnings

from tlm.tlm.tlm_debug_model_fn import model_fn_tlm_debug
from tlm.training import flags_wrapper
from tlm.training.train_config import LMTrainConfig

warnings.filterwarnings("ignore", category=DeprecationWarning)

from my_tf import tf

import tlm.model.base as modeling
from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging
from tlm.model.base import BertModel
from tlm.training.input_fn import input_fn_builder_unmasked
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator
from tlm.training.flags_wrapper import show_input_files


@report_run
def main(_):
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    input_files = flags_wrapper.get_input_files()
    train_config = LMTrainConfig.from_flags(FLAGS)

    show_input_files(input_files)

    model_fn = model_fn_tlm_debug(
        bert_config=bert_config,
        train_config=train_config,
        logging=tf_logging,
        model_class=BertModel,
    )
    if FLAGS.do_predict:
        input_fn = input_fn_builder_unmasked(
            input_files=input_files,
            flags=FLAGS,
            is_training=False)
    else:
        assert False

    r = run_estimator(model_fn, input_fn)
    return r


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
