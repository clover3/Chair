from my_tf import tf
from taskman_client.wrapper import report_run
from tf_util.tf_logging import MuteEnqueueFilter
from tf_util.tf_logging import tf_logging
from tlm.model.mes_hinge import MES_hinge
from tlm.model_cnfig import JsonConfig
from tlm.training.assignment_map import get_init_fn
from tlm.training.flags_wrapper import get_input_files_from_flags, show_input_files
from tlm.training.input_fn import input_fn_builder_pairwise
from tlm.training.model_fn_common import get_tpu_scaffold_or_init, log_var_assignments
from tlm.training.train_config import TrainConfigEx
from tlm.training.train_flags import *
from tlm.training.train_flags import FLAGS
from trainer.tpu_estimator import run_estimator


def metric_fn(log_probs, ):
    return {}


def model_fn_builder(model_config, train_config):
    """Returns `model_fn` closure for TPUEstimator."""
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        tf_logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf_logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = MES_hinge(
            config=model_config,
            is_training=is_training,
            use_one_hot_embeddings=train_config.use_one_hot_embeddings,
            features=features,
        )
        logits = model.get_logits()
        loss = model.get_loss()
        tvars = tf.compat.v1.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if train_config.init_checkpoint:
            initialized_variable_names, init_fn = get_init_fn(train_config, tvars)
            scaffold_fn = get_tpu_scaffold_or_init(init_fn, train_config.use_tpu)
        log_var_assignments(tvars, initialized_variable_names)
        TPUEstimatorSpec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            tvars = None
            train_op = optimization.create_optimizer_from_config(loss, train_config, tvars)
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, train_op=train_op, scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics = (metric_fn, [
                logits
            ])
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, eval_metrics=eval_metrics, scaffold_fn=scaffold_fn)
        else:
            predictions = {
                    "logits": logits,
                    'is_valid_window': model.is_valid_window,
                    'has_any_content': model.has_any_content,
                    'logits3_d': model.logits3_d,
                    'layer1_scores': model.layer1_scores,
                    'max_seg': model.max_seg,
            }

            if "data_id" in features:
                predictions['data_id'] = features['data_id']
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    predictions=predictions,
                    scaffold_fn=scaffold_fn)
        return output_spec
    return model_fn


def run_mes_pairwise():
    input_files = get_input_files_from_flags(FLAGS)
    config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)
    show_input_files(input_files)
    special_flags = FLAGS.special_flags.split(",")
    special_flags.append("feed_features")
    if FLAGS.do_train or FLAGS.do_eval:
        assert False

    else:
        input_fn = input_fn_builder_pairwise(FLAGS.max_d_seq_length, FLAGS)
        model_fn = model_fn_builder(
            config,
            train_config,
        )
    if FLAGS.do_predict:
        tf_logging.addFilter(MuteEnqueueFilter())
    result = run_estimator(model_fn, input_fn)
    return result

@report_run
def main(_):
    return run_mes_pairwise()


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("run_name")
    tf.compat.v1.app.run()
