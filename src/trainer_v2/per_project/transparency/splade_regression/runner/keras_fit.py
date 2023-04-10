import sys

import tensorflow as tf

from misc_lib import path_join
from trainer_v2.train_util.arg_flags import flags_parser

from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.run_config2 import get_run_config2, RunConfig2
from trainer_v2.train_util.get_tpu_strategy import get_strategy2



def get_splade_regression_dataset(run_config) -> tf.data.Dataset:
    text_path = path_join("data", "msmarco", "splade_triplets", "raw.tsv")
    vector_dir = path_join("data", "splade", "splade_encode")
    loader = VectorRegressionLoader(text_path, vector_dir, run_config=run_config)
    int_list = tf.TensorSpec([None, None], dtype=tf.int32)
    vector_sig = tf.TensorSpec(shape=(None, loader.vector_len, ), dtype=tf.float32)
    output_signature = ((int_list, int_list), vector_sig)
    dataset = tf.data.Dataset.from_generator(
        loader.iterate_batched_xy, output_signature=output_signature)
    dataset.batch(1)  # Dummy
    return dataset



@report_run3
def main(args):
    c_log.info("Start {}".format(__file__))
    run_config: RunConfig2 = get_run_config2(args)
    run_config.print_info()
    strategy = get_strategy2(False)
    with strategy.scope():
        new_model = get_regression_model(run_config)
        dataset = get_splade_regression_dataset(run_config)
        max_n = 1000
        line_per_job = 10000
        batch_size = run_config.common_run_config.batch_size

        train_steps = int(max_n * line_per_job / batch_size)
        new_model.fit(dataset, epochs=1, steps_per_epoch=train_steps)



if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
