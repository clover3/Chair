from my_tf import tf
from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging, CounterFilter
from tlm.model.base import BertConfig
from tlm.model.multiple_evidence import MultiEvidenceUseFirst
from tlm.training.classification_model_fn import model_fn_classification
from tlm.training.flags_wrapper import get_input_files_from_flags, show_input_files
from tlm.training.input_fn import input_fn_builder_use_second_input
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

    def override_prediction_fn(predictions, model):
        predictions['vector'] = model.get_output()
        return predictions

    model_fn = model_fn_classification(
        bert_config=bert_config,
        train_config=train_config,
        model_class=MultiEvidenceUseFirst,
        special_flags=special_flags,
        override_prediction_fn=override_prediction_fn
    )
    if FLAGS.do_predict:
        tf_logging.addFilter(CounterFilter())
    input_fn = input_fn_builder_use_second_input(FLAGS)
    result = run_estimator(model_fn, input_fn)
    return result



if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("run_name")
    tf.compat.v1.app.run()
