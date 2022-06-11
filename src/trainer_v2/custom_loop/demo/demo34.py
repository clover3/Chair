import functools
import os
from typing import Iterable

from arg.qck.encode_common import encode_single
from cpath import get_canonical_model_path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.seg_encoder_common import encode_two_segments
from dataset_specific.mnli.mnli_reader import NLIPairData
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.neural_network_def.siamese import ModelConfig200_200

import os

from trainer_v2.custom_loop.demo.demo_common import iterate_and_demo, EncodedSegmentIF, EncodedSegment, \
    enum_hypo_token_tuple, iter_alamri

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from trainer_v2.custom_loop.per_task.nli_ts_util import batch_shaping, load_local_decision_nli


def get_two_seg_asym_encoder():
    model_config = ModelConfig200_200()
    tokenizer = get_tokenizer()
    segment_len = int(model_config.max_seq_length2 / 2)

    def encode_two_seg_input(p_tokens, h_first, h_second):
        input_ids1, input_mask1, segment_ids1 = encode_single(tokenizer, p_tokens, model_config.max_seq_length1)
        triplet2 = encode_two_segments(tokenizer, segment_len, h_first, h_second)
        input_ids2, input_mask2, segment_ids2 = triplet2
        x = input_ids1, segment_ids1, input_ids2, segment_ids2
        x = tuple(map(batch_shaping, x))
        return x

    return encode_two_seg_input


def main():
    c_log.info("Start {}".format(__file__))
    model_path = os.path.join(get_canonical_model_path("nli_ts_run34_0"), "model_25000")

    def get_local_decision_layer_from_model(model):
        local_decision_layer_idx = 12
        local_decision_layer = model.layers[local_decision_layer_idx]
        print("Local decision layer")
        print("Name: ", local_decision_layer.name)
        print("Shape: ", local_decision_layer.output.shape)
        return local_decision_layer

    predictor = load_local_decision_nli(model_path, get_local_decision_layer_from_model)
    tokenizer = get_tokenizer()
    window_size = 3
    encode_two_seg_input = get_two_seg_asym_encoder()

    def enum_hypo_tuples(window_size, e: NLIPairData) -> Iterable[EncodedSegmentIF]:
        p_tokens = tokenizer.tokenize(e.premise)
        for h_first, h_second, st, ed in enum_hypo_token_tuple(tokenizer, e.hypothesis, window_size):
            x = encode_two_seg_input(p_tokens, h_first, h_second)
            yield EncodedSegment(x, p_tokens, [h_first, h_second], st, ed)

    enum_items_by_hypo_seg_enum_fn = functools.partial(enum_hypo_tuples, window_size)
    itr = iter_alamri()
    iterate_and_demo(itr, enum_items_by_hypo_seg_enum_fn, predictor)


if __name__ == "__main__":
    main()
