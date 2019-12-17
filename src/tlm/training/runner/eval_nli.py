import time

import tensorflow as tf

import tlm.model.base as modeling
from taskman_client.task_proxy import get_task_manager_proxy
from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging
from tlm.benchmark.report import save_report
from tlm.model.base import BertModel
from tlm.training.classification_model_fn import model_fn_classification
from tlm.training.input_fn import input_fn_builder_classification as input_fn_builder
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator, TrainConfig, show_input_files


@report_run
def task():
    run_name = FLAGS.output_dir.split("/")[-1]

    begin = time.time()

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))
    train_config = TrainConfig.from_flags(FLAGS)

    show_input_files(input_files)

    model_fn = model_fn_classification(
        bert_config=bert_config,
        train_config=train_config,
        logging=tf_logging,
        model_class=BertModel,
    )
    if FLAGS.do_eval:
        input_fn = input_fn_builder(
            input_files=input_files,
            max_seq_length=FLAGS.max_seq_length,
            is_training=False)
    else:
        raise Exception()

    result = run_estimator(model_fn, input_fn)
    if FLAGS.report_field:
        value = result[FLAGS.report_field]
        proxy = get_task_manager_proxy()
        proxy.report_number(FLAGS.run_name, value)

    save_report("nli", run_name, FLAGS, result["accuracy"])
    print("Elapsed {}".format(time.time() - begin))
    return result


def main(_):
    task()


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
