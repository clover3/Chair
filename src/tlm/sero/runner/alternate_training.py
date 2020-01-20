import tensorflow as tf

from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging
from tlm.model_cnfig import JsonConfig
from tlm.sero.sero_model_fn import model_fn_sero_lm, input_fn_builder
from tlm.training.train_config import LMTrainConfig
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    tf_logging.info("Train Sero Alternate Training")

    config_list = []
    for config_file in FLAGS.model_config_file.split(","):
        config_list.append(JsonConfig.from_json_file(config_file))
    train_config = LMTrainConfig.from_flags(FLAGS)

    is_training = FLAGS.do_train
    input_files_list = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files_list.append(tf.io.gfile.glob(input_pattern))

    # 0 : sero_wiki_2
    # 1 ; Book_corpus
    # 2 : Unmasked_pair
    assert "sero_wiki_2" in input_files_list[0][0]
    assert "sero_3" in input_files_list[1][0]
    assert "sero_pair" in input_files_list[2][0]
    n_task = 3
    batch_size_list = [
        32, 8, 256
    ]
    real_max_train_step= 1000*1000
    steps_per_task = 5000

    while FLAGS.num_train_steps  < real_max_train_step:
        for task_idx in range(0, n_task):
            print("Task {}".format(task_idx))
            input_fn = input_fn_builder(input_files_list[task_idx],
                                        config_list[task_idx].total_sequence_length, FLAGS, is_training)
            model_fn = model_fn_sero_lm(config_list[task_idx], train_config, FLAGS.modeling)
            FLAGS.train_batch_size = batch_size_list[task_idx]
            run_estimator(model_fn, input_fn)
            FLAGS.num_train_steps += steps_per_task



if __name__ == "__main__":
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("modeling")
    tf.compat.v1.app.run()
