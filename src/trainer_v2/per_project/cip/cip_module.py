import tensorflow as tf

from typing import List, Iterable, Callable, Dict, Tuple, Set

from cpath import get_canonical_model_path2
from data_generator.tokenizer_wo_tf import get_tokenizer
from tlm.data_gen.rank_common import join_two_input_ids
from trainer_v2.custom_loop.definitions import ModelConfig300_2
from trainer_v2.custom_loop.inference import InferenceHelper, BERTInferenceHelper
from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config
from trainer_v2.per_project.cip.tfrecord_gen import pad_to_length, join_three_input_ids, join_four_input_ids


def get_cip2(run_config: RunConfig2):
    strategy = get_strategy_from_config(run_config)
    model_config = ModelConfig300_2()
    seq_length = model_config.max_seq_length
    model_save_path = get_canonical_model_path2("nli_cip2_0", "model_3780")
    inf_helper = BERTInferenceHelper(model_save_path, model_config.max_seq_length, strategy)
    tokenizer = get_tokenizer()
    def encode_two(e):
        _, hypo1, hypo2 = e
        hypo1 = tokenizer.convert_tokens_to_ids(hypo1)
        hypo2 = tokenizer.convert_tokens_to_ids(hypo2)
        pair = join_two_input_ids(hypo1, hypo2)
        input_ids = pad_to_length(pair.input_ids, seq_length)
        segment_ids = pad_to_length(pair.seg_ids, seq_length)
        return input_ids, segment_ids

    def predict(items):
        return inf_helper.predict(map(encode_two, items))

    return predict


def get_cip3(run_config: RunConfig2, model_save_path):
    strategy = get_strategy_from_config(run_config)
    model_config = ModelConfig300_2()
    seq_length = model_config.max_seq_length
    inf_helper = BERTInferenceHelper(model_save_path, model_config.max_seq_length, strategy)
    tokenizer = get_tokenizer()

    def encode_three(e):
        _prem, h_full, hypo1, hypo2 = e
        h_full = tokenizer.convert_tokens_to_ids(h_full)
        hypo1 = tokenizer.convert_tokens_to_ids(hypo1)
        hypo2 = tokenizer.convert_tokens_to_ids(hypo2)
        input_ids, segment_ids = join_three_input_ids(h_full, hypo1, hypo2)
        input_ids = pad_to_length(input_ids, seq_length)
        segment_ids = pad_to_length(segment_ids, seq_length)
        return input_ids, segment_ids

    def predict(items):
        ret = inf_helper.predict(list(map(encode_three, items)))
        probs = ret[:, 1].tolist()
        return probs

    return predict


def get_cip_enc_four(run_config: RunConfig2, model_save_path):
    strategy = get_strategy_from_config(run_config)
    model_config = ModelConfig300_2()
    seq_length = model_config.max_seq_length
    inf_helper = BERTInferenceHelper(model_save_path, model_config.max_seq_length, strategy)
    tokenizer = get_tokenizer()

    def encode_four(e):
        ids_list = [tokenizer.convert_tokens_to_ids(tokens) for tokens in e]
        input_ids, segment_ids = join_four_input_ids(ids_list)
        input_ids = pad_to_length(input_ids, seq_length)
        segment_ids = pad_to_length(segment_ids, seq_length)
        return input_ids, segment_ids

    def predict(items):
        ret = inf_helper.predict(list(map(encode_four, items)))
        probs = ret[:, 1].tolist()
        return probs

    return predict
