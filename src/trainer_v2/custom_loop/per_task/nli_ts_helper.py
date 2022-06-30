import os
from typing import Callable, List, Iterable, Tuple

from cpath import common_model_dir_root
from taskman_client.task_proxy import get_local_machine_name
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.per_task.nli_ts_util import get_two_seg_concat_encoder, get_two_seg_asym_encoder, \
    load_local_decision_model, LocalDecisionNLICore
from trainer_v2.train_util.get_tpu_strategy import get_strategy_by_machine_name

EncoderType = Callable[[List, List, List], Iterable[Tuple]]


def get_encode_fn(encoder_name, model) -> EncoderType:
    if encoder_name == "concat":
        max_seg_length1 = model.inputs[0].shape[1]
        c_log.info("Using get_two_seg_concat_encoder")
        encode_fn: EncoderType = get_two_seg_concat_encoder(max_seg_length1)
    elif encoder_name == "two_seg":
        max_seg_length1 = model.inputs[0].shape[1]
        max_seg_length2 = model.inputs[2].shape[1]
        c_log.info("Using get_two_seg_asym_encoder")
        encode_fn: EncoderType = get_two_seg_asym_encoder(max_seg_length1, max_seg_length2, False)
    else:
        raise ValueError(encoder_name)
    return encode_fn


def get_local_decision_nlits_core(run_name, encoder_name):
    strategy = get_strategy_by_machine_name()
    model_path = get_model_path(run_name)
    with strategy.scope():
        c_log.debug("Loading model from {} ...".format(model_path))
        model = load_local_decision_model(model_path)
        encode_fn = get_encode_fn(encoder_name, model)
        c_log.debug("Done")
        nlits: LocalDecisionNLICore = LocalDecisionNLICore(model, strategy, encode_fn)
    return nlits


def get_model_path(model_name):
    machine_name = get_local_machine_name()
    if machine_name == "us-1":
        model_path = f'gs://clovertpu/training/model/{model_name}'
    elif machine_name == "GOSFORD":
        model_path = os.path.join(common_model_dir_root, 'runs', f"{model_name}")
    elif machine_name == "ingham.cs.umass.edu":
        model_path = os.path.join(common_model_dir_root, 'runs', f"{model_name}")
    else:
        # It should be tpu v4
        model_path = f'/home/youngwookim/code/Chair/model/{model_name}'
    return model_path