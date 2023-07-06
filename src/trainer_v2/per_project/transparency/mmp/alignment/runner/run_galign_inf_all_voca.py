import os
import pickle
from dataclasses import dataclass

from keras import Input, Model
from keras.layers import Dense

from data_generator.tokenizer_wo_tf import get_tokenizer
from trainer_v2.custom_loop.modeling_common.tf_helper import distribute_dataset
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config
from trainer_v2.per_project.transparency.mmp.alignment.dataset_factory import read_galign, read_galign_v2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from trainer_v2.chair_logging import c_log
import tensorflow as tf
from taskman_client.wrapper3 import report_run3
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2, get_run_config_for_predict
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def main(args):
    c_log.info(__file__)
    run_config: RunConfig2 = get_run_config_for_predict(args)
    run_config.print_info()
    tokenizer = get_tokenizer()

    def build_dataset(q_term: int, d_term_id_st: int, d_term_id_ed: int):
        # Create a range of integers from st to ed
        data_range = tf.range(d_term_id_st, d_term_id_ed)

        # Create a tf.data.Dataset from tensor slices
        dataset = tf.data.Dataset.from_tensor_slices({
            'd_term': data_range
        })
        def add_q_term_make_array(x):
            x = {'d_term': [x['d_term']],
                 'q_term': tf.constant([q_term], dtype=tf.int32),
                 'raw_label': tf.zeros([1], dtype=tf.float32),
                 'label': tf.zeros([1], dtype=tf.int32),
                 'is_valid': tf.zeros([1], dtype=tf.int32),
                 }
            return x
        # Add the q_term feature to each record in the dataset
        dataset = dataset.map(add_q_term_make_array)
        return dataset

    strategy = get_strategy_from_config(run_config)
    d_term_id_st = 1997
    d_term_id_ed = 10000
    with strategy.scope():
        while True:
            q_term = input("Enter query term: ")
            q_term_id = tokenizer.convert_tokens_to_ids([q_term])[0]
            eval_dataset = build_dataset(q_term_id, d_term_id_st, d_term_id_ed)
            batched_dataset = eval_dataset.batch(run_config.common_run_config.batch_size)
            model = tf.keras.models.load_model(run_config.predict_config.model_save_path, compile=False)
            outputs = model.predict(batched_dataset)
            scores = outputs['align_probe']['all_concat']
            preds = tf.less(0, scores)
            print("q_term", q_term,)
            pos = []
            for i, record in enumerate(eval_dataset):
                pred = preds[i].numpy()
                d_term = record['d_term'][0].numpy().tolist()
                if pred:
                    pos.append(d_term)

            print(tokenizer.convert_ids_to_tokens(pos))

if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


