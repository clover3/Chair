from collections import OrderedDict
from typing import List, Iterable, Callable, Dict, Tuple, Set

from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.es_mmp.common import concat_ph_to_encode_fn
from data_generator2.segmented_enc.es_nli.common import PHSegmentedPair
from dataset_specific.mnli.mnli_reader import NLIPairData
from trainer_v2.per_project.cip.cip_common import get_random_split_location
from trainer_v2.chair_logging import c_log
import tensorflow as tf


def partition_query(
        tokenizer, qd_pair: Tuple[str, str]):
    query, document = qd_pair
    q_tokens = tokenizer.tokenize(query)
    d_tokens = tokenizer.tokenize(document)
    h_st, h_ed = get_random_split_location(q_tokens)
    nli_pair = NLIPairData(document, query, "0", "0")
    ph_seg_pair = PHSegmentedPair(d_tokens, q_tokens, h_st, h_ed, [], [], nli_pair)
    return ph_seg_pair


def get_pep_scorer(model_path, batch_size=16) -> Callable[[List[Tuple[str, str]]], Iterable[float]]:
    segment_len = 256
    max_seq_length = segment_len * 2
    c_log.info("Loading model from %s", model_path)
    pointwise_model = tf.keras.models.load_model(model_path, compile=False)
    inference_model = pointwise_model
    tokenizer = get_tokenizer()
    SpecI = tf.TensorSpec([max_seq_length], dtype=tf.int32)
    sig = (SpecI, SpecI,),

    def score_fn(qd_list: List):
        def generator():
            for qd in qd_list:
                e = partition_query(tokenizer, qd)
                input_ids, input_mask, segment_ids = concat_ph_to_encode_fn(tokenizer, segment_len, e)
                yield (input_ids, segment_ids),

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=sig)
        dataset = dataset.batch(batch_size)
        output = inference_model.predict(dataset)
        return output[:, 1]

    c_log.info("Defining network")
    return score_fn
