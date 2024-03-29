from my_tf import tf
from taskman_client.wrapper import report_run
from tlm.training.input_fn_var_length import input_fn_builder, extract_record_fn
from tlm.training.ranking_model_fn import model_fn_ranking
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator

@report_run
def main(_):
    input_fn = input_fn_builder(FLAGS, extract_record_fn)
    model_fn = model_fn_ranking(FLAGS)
    result = run_estimator(model_fn, input_fn)
    return result


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("run_name")
    tf.compat.v1.app.run()

