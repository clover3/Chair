import tensorflow as tf

from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging, MuteEnqueueFilter
from tlm.model.triple_bert_masking import TripleBertMaskingRelevanceTuning
from tlm.model_cnfig import JsonConfig
from tlm.training.classification_model_fn import model_fn_classification
from tlm.training.input_fn import input_fn_builder_cppnc_triple
from tlm.training.train_config import TrainConfigEx
from tlm.training.train_flags import *
from tlm.training.train_flags import FLAGS
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    tf_logging.info("TripleBertMasking")
    config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)
    input_fn = input_fn_builder_cppnc_triple(FLAGS)
    special_flags = FLAGS.special_flags.split(",")
    special_flags.append("feed_features")
    special_flags.append("ask_tvar")

    def override_prediction_fn(predictions, model):
        for key, value in model.get_predictions().items():
            predictions[key] = value
        return predictions

    model_fn = model_fn_classification(config, train_config, TripleBertMaskingRelevanceTuning,
                                       special_flags, override_prediction_fn)
    if FLAGS.do_predict:
        tf_logging.addFilter(MuteEnqueueFilter())

    return run_estimator(model_fn, input_fn)


if __name__ == "__main__":
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()

