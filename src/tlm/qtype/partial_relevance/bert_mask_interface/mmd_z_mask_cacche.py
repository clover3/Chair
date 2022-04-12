from typing import List, Callable, Dict, Tuple

from bert_api.segmented_instance.seg_instance import SegmentedInstance
from cpath import at_output_dir
from datastore.sql_based_cache_client import SQLBasedCacheClientS
from bert_api.bert_masking_common import BERTMaskIF
from bert_api.task_clients.mmd_z_interface.mmd_z_mask_client import get_mmd_z_mask_client
from alignment.data_structure.eval_data_structure import get_test_segment_instance
from bert_api.attn_mask_utils import BertMaskWrap

InputType = Tuple[SegmentedInstance, Dict]
OutputType = List[float]
AttnMaskForward = Callable[[List[InputType]], List[OutputType]]


def get_mmd_z_mask_cache_client(option) -> SQLBasedCacheClientS[InputType, OutputType]:
    raw_client: BERTMaskIF = get_mmd_z_mask_client(option)
    core = BertMaskWrap(raw_client, max_seq_length=512)
    forward_fn: Callable[[List[InputType]], List[OutputType]] = core.eval
    cache_path = at_output_dir("qtype", "mmd_z_mask_cache.sqlite")

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
    cache_client = get_mmd_z_mask_cache_client(option)
    return cache_client.predict


def test_save():
    segment_instance: SegmentedInstance = get_test_segment_instance()
    cache_client = get_mmd_z_mask_cache_client("localhost")
    items = [(segment_instance, {})]
    cache_client.predict(items)


def main():
    test_save()


if __name__ == "__main__":
    main()

