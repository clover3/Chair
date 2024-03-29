from misc_lib import ceil_divide
from trainer_v2.custom_loop.definitions import ModelConfig600_3
from trainer_v2.custom_loop.per_task.nli_ts_util import load_local_decision_model_n_label_3
from trainer_v2.custom_loop.run_config2 import get_eval_run_config2
import sys
import traceback

from port_info import LOCAL_DECISION_PORT
from rpc.bert_like_server import RPCServerWrap
from typing import List, Callable, Iterable, Dict, Tuple, NamedTuple
import tensorflow as tf

from trainer_v2.custom_loop.inference import InferenceHelper
from trainer_v2.custom_loop.per_task.nli_ts_util import load_local_decision_model_n_label_3
from trainer_v2.custom_loop.run_config2 import get_eval_run_config2
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config
from trainer_v2.train_util.arg_flags import flags_parser


def run_server(args):
    run_config = get_eval_run_config2(args)
    strategy = get_strategy_from_config(run_config)

    def model_factory():
        model: tf.keras.models.Model = load_local_decision_model_n_label_3(run_config.get_model_path())
        return model

    model_config = ModelConfig600_3()

    def dataset_factory(payload: List):
        def generator():
            print(payload[0])
            yield from payload

        int_list = tf.TensorSpec(shape=(model_config.max_seq_length,), dtype=tf.int32)
        output_signature = [int_list, int_list]
        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        return dataset

    predictor = InferenceHelper(model_factory, dataset_factory, strategy)

    def predict(payload: List[Tuple[List[int], List[int]]]):
        try:
            # TODO      cast output to list (serializable)
            stacked_output = predictor.predict(payload)
            l_decisions, g_decision_l = stacked_output
            print(l_decisions.shape)
            print(g_decision_l[0].shape)
            print(type(l_decisions))

            output = []
            for i in range(len(payload)):
                output.append((l_decisions[i].tolist(), g_decision_l[0].tolist()))
            return output
        except Exception as e:
            print("Exception in user code:")
            print(traceback.print_exc(file=sys.stdout))
            print(e)
        return []


    item = [0 for _ in range(model_config.max_seq_length)]
    payload = [(item, item)] * 4
    output = predict(payload)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    run_server(args)
