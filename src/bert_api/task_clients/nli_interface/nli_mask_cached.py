from typing import List, Callable, Dict, Tuple
from cpath import at_output_dir

from bert_api.segmented_instance.seg_instance import SegmentedInstance
from bert_api.task_clients.bert_masking_client import get_localhost_bert_mask_client
from bert_api.bert_masking_common import BERTMaskIF

from datastore.sql_based_cache_client import SQLBasedCacheClientS
from bert_api.attn_mask_utils import BertMaskWrap
from contradiction.alignment.data_structure.eval_data_structure import get_test_segment_instance

InputType = Tuple[SegmentedInstance, Dict]
OutputType = List[float]
AttnMaskForward = Callable[[List[InputType]], List[OutputType]]


def get_nli_mask_cache_client(option) -> SQLBasedCacheClientS[InputType, OutputType]:
    if option != "localhost":
        raise ValueError()
    predictor: BERTMaskIF = get_localhost_bert_mask_client()
    core = BertMaskWrap(predictor, max_seq_length=300)
    forward_fn: Callable[[List[InputType]], List[OutputType]] = core.eval
    cache_path = at_output_dir("nli", "nli_mask_cache.sqlite")

    def hash_fn(item: InputType) -> str:
        seg, d = item
        s1 = SegmentedInstance.str_hash(seg)
        return s1 + str(d)

    cache_client: SQLBasedCacheClientS[InputType, OutputType] = \
        SQLBasedCacheClientS(forward_fn,
                             hash_fn,
                             0.035,
                             cache_path,
                             1)
    return cache_client


def get_attn_mask_forward_fn(option: str) -> AttnMaskForward:
    cache_client = get_nli_mask_cache_client(option)
    return cache_client.predict


def test_save():
    segment_instance: SegmentedInstance = get_test_segment_instance()
    cache_client = get_nli_mask_cache_client("localhost")
    items = [(segment_instance, {})]
    cache_client.predict(items)


def main():
    test_save()


if __name__ == "__main__":
    main()

