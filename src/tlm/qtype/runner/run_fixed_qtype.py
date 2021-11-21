from my_tf import tf
from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging, MuteEnqueueFilter
from tlm.qtype.fixed_qtype_model_fn import model_fn_fixed_qtype, qtype_id_modeling_identity, process_feature
from tlm.qtype.qde_model_fn import get_mae_loss_modeling, qtype_modeling_single_mlp
from tlm.qtype.qtype_input_fn import input_fn_builder_fixed_qtype
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    input_fn = input_fn_builder_fixed_qtype(max_seq_length=FLAGS.max_seq_length, flags=FLAGS)
    model_fn = model_fn_fixed_qtype(FLAGS,
                                    process_feature,
                                    get_mae_loss_modeling,
                                    qtype_id_modeling_identity,
                                    qtype_modeling_single_mlp,
                                    )
    if FLAGS.do_predict:
        tf_logging.addFilter(MuteEnqueueFilter())

    return run_estimator(model_fn, input_fn)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("run_name")
    tf.compat.v1.app.run()
