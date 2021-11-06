from my_tf import tf
from taskman_client.wrapper import report_run
from tlm.qtype.BertQType import BertQWordPred
from tlm.qtype.qtype_input_fn import input_fn_builder_qtype
from tlm.qtype.qtype_model_fn import model_fn_qtype_pairwise
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    input_fn = input_fn_builder_qtype(FLAGS.max_seq_length, FLAGS)
    model_fn = model_fn_qtype_pairwise(FLAGS, BertQWordPred)
    return run_estimator(model_fn, input_fn)


if __name__ == "__main__":
    model_config_flag_checking()
    tf.compat.v1.app.run()
