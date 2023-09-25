import sys
import traceback
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from port_info import LOCAL_DECISION_PORT
from rpc.bert_like_server import RPCServerWrap
from trainer_v2.custom_loop.definitions import ModelConfigType
from trainer_v2.custom_loop.inference import InferenceHelper, SanityChecker
from trainer_v2.custom_loop.per_task.ts_util import get_dataset_factory_two_seg, load_local_decision_model


def run_local_decision_server(run_config, model_config:ModelConfigType, strategy):
    def model_factory():
        model: tf.keras.models.Model = load_local_decision_model(
            model_config, run_config.get_model_path())
        return model

    dataset_factory = get_dataset_factory_two_seg(model_config)
    predictor = InferenceHelper(model_factory, dataset_factory, strategy)
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