import sys

import tensorflow as tf
from transformers import TFMPNetModel, MPNetConfig, MPNetTokenizer

from cache import load_from_pickle
from trainer_v2.arg_flags import flags_parser
from trainer_v2.get_tpu_strategy import get_strategy


def load_reference_embedding():
    return load_from_pickle("hello_world_sent_transformer")


class TFSBERT:
    def __init__(self, model_path=None, config_path=None):
        self.tokenizer = MPNetTokenizer.from_pretrained("microsoft/mpnet-base")
        self.model = load_tf_mp_net_model(config_path, model_path)
        print("TFSBERT loaded")

    def predict(self, text_list):
        input = self.tokenizer(text_list, return_tensors="tf", padding=True)
        return self.predict_from_ids(input)

    def predict_from_ids(self, input):
        outputs = self.model(input)
        last_hidden = outputs.last_hidden_state
        input_mask = input["attention_mask"]
        input_mask_f = tf.expand_dims(tf.cast(input_mask, tf.float32), axis=2)
        sum_vectors = tf.reduce_sum(last_hidden * input_mask_f, axis=1)
        num_tokens = tf.reduce_sum(input_mask_f, axis=1)
        avg_vectors = tf.divide(sum_vectors, num_tokens)
        return avg_vectors


def main(args):
    strategy = get_strategy(True, "v2-2")
    with strategy.scope():
        sbert = TFSBERT(args.init_checkpoint, args.config_path)
        print("Model loaded")
        avg_vectors = sbert.predict(["ValueError: Can't convert non-rectangular Python sequence to Tensor.",
                        "During handling of the above exception, another exception occurred:"])
        print("DONE")


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


def load_tf_mp_net_model(config_path, convert_save_path):
    configuration = MPNetConfig.from_json_file(config_path)
    model = TFMPNetModel(configuration)
    model.compile()
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(convert_save_path).assert_existing_objects_matched()
    return model