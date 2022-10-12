import os
from typing import Callable, List, Iterable, Tuple

from cpath import common_model_dir_root
from taskman_client.task_proxy import get_local_machine_name
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.per_task.nli_ts_util import get_two_seg_concat_encoder, get_two_seg_asym_encoder, \
    load_local_decision_model, LocalDecisionNLICore, load_local_decision_model_as_second_only, \
    LocalDecisionNLICoreSecond, get_concat_mask_encoder
from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config

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
    elif encoder_name == "concat_wmask":
        max_seg_length1 = model.inputs[0].shape[1]
        c_log.info("Using get_two_seg_concat_encoder")
        encode_fn: EncoderType = get_concat_mask_encoder(max_seg_length1)
    else:
        raise ValueError(encoder_name)
    return encode_fn


def get_local_decision_nlits_core(run_config: RunConfig2, encoder_name):
    model_path = run_config.get_model_path()
    strategy = get_strategy_from_config(run_config)
    with strategy.scope():
        c_log.debug("Loading model from {} ...".format(model_path))
        model = load_local_decision_model(model_path)
        encode_fn = get_encode_fn(encoder_name, model)
        c_log.debug("Done")
        nlits: LocalDecisionNLICore \
            = LocalDecisionNLICore(model,
                                   strategy,
                                   encode_fn,
                                   run_config.common_run_config.batch_size)
    return nlits


def get_local_decision_nlits_core2(run_config, encoder_name):
    model_path = run_config.get_model_path()
    strategy = get_strategy_from_config(run_config)
    with strategy.scope():
        c_log.debug("Loading model from {} ...".format(model_path))
        model = load_local_decision_model_as_second_only(model_path)
        encode_fn = get_encode_fn(encoder_name, model)
        c_log.debug("Done")
        nlits: LocalDecisionNLICore = LocalDecisionNLICoreSecond(model, strategy, encode_fn)
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