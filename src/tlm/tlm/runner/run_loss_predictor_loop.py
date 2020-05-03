import os
import pickle

import tensorflow as tf

import tlm.model.base as modeling
from sydney_manager import MarkedTaskManager
from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging, CounterFilter
from tlm.estimator_loop import run_estimator_loop
from tlm.model.base import BertModel
from tlm.model_cnfig import JsonConfig
from tlm.tlm.loss_diff_prediction_model import loss_diff_predict_only_model_fn
from tlm.training.input_fn import input_fn_builder_unmasked
from tlm.training.train_config import TrainConfigEx
from tlm.training.train_flags import *


class Worker:
    def __init__(self, out_dir, input_gs_dir):
        self.output_dir = out_dir
        self.input_gs_dir = input_gs_dir
        bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
        train_config = TrainConfigEx.from_flags(FLAGS)
        model_config = JsonConfig.from_json_file(FLAGS.model_config_file)
        tf_logging.addFilter(CounterFilter())

        model_fn = loss_diff_predict_only_model_fn(
            bert_config=bert_config,
            train_config=train_config,
            model_class=BertModel,
            model_config=model_config,
        )

        tf.io.gfile.makedirs(FLAGS.output_dir)

        tpu_cluster_resolver = None
        if FLAGS.use_tpu and FLAGS.tpu_name:
            tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
                FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
        config = tf.compat.v1.ConfigProto(allow_soft_placement=False, )
        is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.compat.v1.estimator.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=FLAGS.master,
            model_dir=FLAGS.output_dir,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours,
            session_config=config,
            tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
                iterations_per_loop=FLAGS.iterations_per_loop,
                num_shards=FLAGS.num_tpu_cores,
                per_host_input_for_training=is_per_host))

        # If TPU is not available, this will fall back to normal Estimator on CPU
        # or GPU.
        self.estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=FLAGS.train_batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            predict_batch_size=FLAGS.eval_batch_size,
        )

    def work(self, job_id):
        output_name = os.path.join(self.output_dir, str(job_id))
        tf_logging.info("Predicting for %s", output_name)
        input_file = self.input_gs_dir + "/" + str(job_id)
        input_fn = input_fn_builder_unmasked(
            input_files=[input_file],
            flags=FLAGS,
            is_training=False)

        result = self.estimator.predict(input_fn=input_fn, yield_single_examples=False)
        pickle.dump(list(result), open(output_name, "wb"))


# RLPP.sh
@report_run
def main(_):
    new_main()


def new_main():
    working_path = "disk_output"
    mark_path = os.path.join(working_path, "loss_predictor_predictions_mark")
    out_path = os.path.join(working_path, "loss_predictor_predictions")
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    gs_input_dir = "gs://clovertpu/training/data/unmasked_pair_x3"
    mtm = MarkedTaskManager(4000, mark_path, 1)
    worker = Worker(out_path, gs_input_dir)
    job_id = mtm.pool_job()
    print("Job id : ", job_id)
    while job_id is not None:
        worker.work(job_id)
        job_id = mtm.pool_job()
        print("Job id : ", job_id)


def old_main():
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))
    train_config = TrainConfigEx.from_flags(FLAGS)
    model_config = JsonConfig.from_json_file(FLAGS.model_config_file)
    tf_logging.addFilter(CounterFilter())

    output_names = []
    input_fn_list = []
    for input_file in input_files:
        name = input_file.split("/")[-1]
        output_name = "disk_output/loss_predictor_predictions/" + name
        input_fn = input_fn_builder_unmasked(
            input_files=[input_file],
            flags=FLAGS,
            is_training=False)
        input_fn_list.append(input_fn)
        output_names.append(output_name)

    model_fn = loss_diff_predict_only_model_fn(
        bert_config=bert_config,
        train_config=train_config,
        model_class=BertModel,
        model_config=model_config,
    )
    if FLAGS.do_predict:
        run_estimator_loop(model_fn, input_fn_list, output_names)
    else:
        raise Exception("Only PREDICT mode is allowed")


if __name__ == "__main__":
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
