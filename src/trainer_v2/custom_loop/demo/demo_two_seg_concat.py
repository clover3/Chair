import os
import sys
from typing import Iterable

from cpath import get_canonical_model_path
from trainer_v2.custom_loop.demo.demo_common import iterate_and_demo, EncodedSegmentIF, EncodedSegment, \
    enum_hypo_token_tuple, iter_alamri

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.mnli.mnli_reader import NLIPairData
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.per_task.nli_ts_util import load_local_decision_nli_model, get_two_seg_concat_encoder


def main():
    c_log.info("Start {}".format(__file__))
    model_path = os.path.join(get_canonical_model_path("nli_ts_run40_0"), "model_25000")
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    def get_local_decision_layer_from_model(model):
        local_decision_layer_idx = 17
        local_decision_layer = model.layers[local_decision_layer_idx]
        for idx, layer in enumerate(model.layers):
            print("layer {}: {} {}".format(idx, layer.name, layer.output.shape))
        print("Local decision layer")
        print("Name: ", local_decision_layer.name)
        print("Shape: ", local_decision_layer.output.shape)
        return local_decision_layer

    c_log.info("Loading model from %s", model_path)
    predictor = load_local_decision_nli_model(model_path, get_local_decision_layer_from_model)
    tokenizer = get_tokenizer()
    window_size = 3
    encode_two_seg_input = get_two_seg_concat_encoder()

    def enum_hypo_tuples(e: NLIPairData, window_size) -> Iterable[EncodedSegmentIF]:
        p_tokens = tokenizer.tokenize(e.premise)
        for h_first, h_second in enum_hypo_token_tuple(tokenizer, e.hypothesis, window_size):
            x = encode_two_seg_input(p_tokens, h_first, h_second)
            yield EncodedSegment(x, p_tokens, [h_first, h_second])

    def enum_hypo_tuples_fn(e: NLIPairData):
        return enum_hypo_tuples(e, window_size)
    itr = iter_alamri()

    iterate_and_demo(itr, enum_hypo_tuples_fn, predictor)


if __name__ == "__main__":
    main()
