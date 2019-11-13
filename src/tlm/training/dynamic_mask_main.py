from tlm.tf_logging import tf_logging, logging
import tensorflow as tf

from tlm.training.train_flags import *
import tlm.model.base as modeling
from tlm.training.input_fn import input_fn_builder_unmasked as input_fn_builder
from tlm.training.model_fn import model_fn_random_masking, model_fn_target_masking
from tlm.model.base import BertModel

import os

class TrainConfig:
    def __init__(self,
                 init_checkpoint,
                 learning_rate,
                 num_train_steps,
                 num_warmup_steps,
                 use_tpu,
                 use_one_hot_embeddings,
                 max_predictions_per_seq
                 ):
        self.init_checkpoint = init_checkpoint
        self.learning_rate = learning_rate
        self.num_train_steps = num_train_steps
        self.num_warmup_steps = num_warmup_steps
        self.use_tpu = use_tpu
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.max_predictions_per_seq = max_predictions_per_seq

    @classmethod
    def from_flags(cls, flags):
        return TrainConfig(
            flags.init_checkpoint,
            flags.learning_rate,
            flags.num_train_steps,
            flags.num_warmup_steps,
            flags.use_tpu,
            flags.use_tpu,
            flags.max_predictions_per_seq
        )

def main(_):
    lm_pretrain()

def lm_pretrain():
    tf_logging.setLevel(logging.INFO)
    if FLAGS.log_debug:
        tf_logging.setLevel(logging.DEBUG)


    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    tf.io.gfile.makedirs(FLAGS.output_dir)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))

    tf_logging.info("*** Input Files ***")
    for idx, input_file in enumerate(input_files):
        tf_logging.info("  %s" % input_file)
        if idx > 10 :
          break
    tf_logging.info("Total of %d files" % len(input_files))

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    config = tf.compat.v1.ConfigProto(allow_soft_placement=False,)
    is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.compat.v1.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_every_n_hours =FLAGS.keep_checkpoint_every_n_hours,
      session_config=config,
      tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

    model_fn_builder = model_fn_random_masking
    train_config = TrainConfig.from_flags(FLAGS)

    if FLAGS.target_lm:
      model_fn_builder = model_fn_target_masking
      train_config.target_task_checkpoint = FLAGS.target_task_checkpoint

    model_fn = model_fn_builder(
        bert_config=bert_config,
        train_config=train_config,
        logging=tf_logging,
        model_class=BertModel,
    )

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.eval_batch_size,
    )

    if FLAGS.do_train:
        tf_logging.info("***** Running training *****")
        tf_logging.info("  Batch size = %d", FLAGS.train_batch_size)
        train_input_fn = input_fn_builder(
            input_files=input_files,
            max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=True)
        estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)

    if FLAGS.do_eval:
        tf_logging.info("***** Running evaluation *****")
        tf_logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        eval_input_fn = input_fn_builder(
            input_files=input_files,
            max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=False)

        result = estimator.evaluate(
            input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.io.gfile.GFile(output_eval_file, "w") as writer:
          tf_logging.info("***** Eval results *****")
          for key in sorted(result.keys()):
            tf_logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
