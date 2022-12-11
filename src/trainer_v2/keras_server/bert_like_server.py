import sys
import traceback
from typing import List, Tuple

import tensorflow as tf

from rpc.bert_like_server import RPCServerWrap
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.inference import InferenceHelper
from trainer_v2.custom_loop.train_loop import load_model_by_dir_or_abs

KERAS_BERT_LIKE_INPUT_SIG = List[Tuple[List[int], List[int]]]


def run_keras_bert_like_server(port, model_path, model_config, strategy):
    def model_factory():
        return load_model_by_dir_or_abs(model_path)

    def dataset_factory(payload: List):
        def generator():
            for item in payload:
                yield tuple(item)

        int_list = tf.TensorSpec(shape=(model_config.max_seq_length,), dtype=tf.int32)
        output_signature = (int_list, int_list)
        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        return dataset

    predictor = InferenceHelper(model_factory, dataset_factory, strategy)

    def predict(payload: KERAS_BERT_LIKE_INPUT_SIG):
        try:
            outputs = predictor.predict(payload)
            output = []
            for i in range(len(payload)):
                output.append((outputs[i].tolist()))
            return output

        except Exception as e:
            c_log.warn("Exception in user code:")
            c_log.warn(traceback.print_exc(file=sys.stdout))
            c_log.warn(e)
        return []

    server = RPCServerWrap(predict)
    c_log.info("server started")
    server.start(port)
