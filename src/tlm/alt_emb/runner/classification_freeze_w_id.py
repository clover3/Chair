
from my_tf import tf

from taskman_client.wrapper import report_run
from tlm.model.base import BertConfig
from tlm.model.freeze_bert import FreezeEmbedding
from tlm.training.classification_model_fn import model_fn_classification
from tlm.training.flags_wrapper import get_input_files_from_flags, show_input_files
from tlm.training.input_fn import input_fn_builder_classification_w_data_id
from tlm.training.train_config import TrainConfigEx
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    input_files = get_input_files_from_flags(FLAGS)
    bert_config = BertConfig.from_json_file(FLAGS.bert_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)
    show_input_files(input_files)
    special_flags = FLAGS.special_flags.split(",")
    model_fn = model_fn_classification(
        bert_config=bert_config,
        train_config=train_config,
        model_class=FreezeEmbedding,
        special_flags=special_flags,
    )

    input_fn = input_fn_builder_classification_w_data_id(
        input_files=input_files,
        flags=FLAGS,
        is_training=FLAGS.do_train)

    result = run_estimator(model_fn, input_fn)
    return result


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("run_name")
    tf.compat.v1.app.run()
