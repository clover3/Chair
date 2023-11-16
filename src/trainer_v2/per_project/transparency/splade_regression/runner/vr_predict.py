import sys
import tensorflow as tf


from misc_lib import write_to_lines
from taskman_client import wrapper3
from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2, get_run_config_for_predict
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config
from trainer_v2.per_project.transparency.splade_regression.data_loaders.dataset_factories import get_three_text_dataset, \
    get_text_pair_dataset, get_text_pair_dataset2
from trainer_v2.train_util.arg_flags import flags_parser


def get_qd_scorer_model(encoder_model, max_seq_len) -> tf.keras.models.Model:
    q_input_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype='int64', name=f"input_ids_0")
    q_attention_mask = tf.keras.layers.Input(shape=(max_seq_len,), dtype='int64', name=f"attention_mask_0")
    d_input_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype='int64', name=f"input_ids_1")
    d_attention_mask = tf.keras.layers.Input(shape=(max_seq_len,), dtype='int64', name=f"attention_mask_1")
    inputs = [q_input_ids, q_attention_mask, d_input_ids, d_attention_mask]

    def encode(input_ids, attention_mask):
        t_input = (input_ids, attention_mask)
        return encoder_model((t_input,))

    # q_enc = encode(q_input_ids, q_attention_mask)
    # d_enc = encode(d_input_ids, d_attention_mask)
    t_input = (q_input_ids, q_attention_mask)
    q_enc = encoder_model((t_input, ))
    t_input = (d_input_ids, d_attention_mask)
    d_enc = encoder_model((t_input, ))

    t = tf.multiply(q_enc, d_enc)
    score = tf.reduce_sum(t, axis=1)
    print(score)
    outputs = score
    print("This model returns  score ")
    return tf.keras.models.Model(inputs=inputs, outputs=outputs)


@report_run3
def main(args):
    c_log.info(__file__)
    run_config: RunConfig2 = get_run_config_for_predict(args)

    model_config = {
        "max_seq_length": 256
    }
    max_seq_len = model_config["max_seq_length"]
    strategy = get_strategy_from_config(run_config)
    with strategy.scope():
        dataset: tf.data.Dataset = get_text_pair_dataset2(
            run_config.dataset_config.eval_files_path, model_config, run_config,
            is_for_training=False
        )
        c_log.info("loading model from %s", run_config.predict_config.model_save_path)
        encoder_model = tf.keras.models.load_model(run_config.predict_config.model_save_path)
        model = get_qd_scorer_model(encoder_model, max_seq_len)
        prediction = model.predict(dataset)

    print(prediction.shape)
    print(type(prediction[0]))
    write_to_lines(prediction, run_config.predict_config.predict_save_path)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
