import sys
import traceback
from collections import Counter
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from port_info import LOCAL_DECISION_PORT
from rpc.bert_like_server import RPCServerWrap
from trainer_v2.custom_loop.inference import InferenceHelper
from trainer_v2.custom_loop.per_task.nli_ts_util import load_local_decision_model, dataset_factory_600_3
from trainer_v2.custom_loop.run_config2 import get_eval_run_config2
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config
from trainer_v2.train_util.arg_flags import flags_parser


class SanityChecker:
    def __init__(self):
        print("Using basic SanityChecker")
        self.window = []

    def update(self, labels):
        self.window.extend(labels)
        n_label_diverse = len(Counter(self.window))
        if n_label_diverse == 1 and len(self.window) > 100:
            print("[WARNING] There were {} predictions which were all {}".format(len(self.window), n_label_diverse))

        self.window = self.window[:100]


def run_server(args):
    run_config = get_eval_run_config2(args)
    strategy = get_strategy_from_config(run_config)

    def model_factory():
        model: tf.keras.models.Model = load_local_decision_model(run_config.get_model_path())
        return model
    predictor = InferenceHelper(model_factory, dataset_factory_600_3, strategy)
    sanity_checker = SanityChecker()

    def predict(payload: List[Tuple[List[int], List[int]]]) -> List[Tuple[List[List[float]], List[float]]]:
        try:
            stacked_output = predictor.predict(payload)
            l_decisions, g_decision_l = stacked_output
            l_labels = np.reshape(np.argmax(l_decisions, axis=2), [-1])
            sanity_checker.update(l_labels)
            output = []
            for i in range(len(payload)):
                output.append((l_decisions[i].tolist(), g_decision_l[0][i].tolist()))
            return output

        except Exception as e:
            print("Exception in user code:")
            print(traceback.print_exc(file=sys.stdout))
            print(e)
        return []

    server = RPCServerWrap(predict)
    print("server started")
    server.start(LOCAL_DECISION_PORT)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    run_server(args)
