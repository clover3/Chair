from typing import Iterable, Tuple, List

from transformers import AutoTokenizer

from data_generator2.segmented_enc.es_common.es_two_seg_common import Segment1PartitionedPair, RangePartitionedSegment, \
    PairData
from dataset_specific.msmarco.passage.path_helper import iter_train_triples_partition
from dataset_specific.msmarco.passage.processed_resource_loader import load_partitioned_query
from list_lib import lmap, lflatten
from trainer_v2.per_project.transparency.mmp.pep.seg_enum_helper import translate_sp_idx_to_sb_idx


def iter_qd(part_no) -> Iterable[Segment1PartitionedPair]:
    tokenizer2 = AutoTokenizer.from_pretrained("bert-base-uncased")

    triplet_iter = iter_train_triples_partition(part_no)
    query_st_ed_iter: Iterable[Tuple[List[str], int, int]] = load_partitioned_query(part_no)

    data_idx = part_no * 1000000
    error_cnt = 0
    for (q, d1, d2), (q_sp_tokens, st, ed) in zip(triplet_iter, query_st_ed_iter):
        # check q equals q_tokens
        if data_idx % 17 == 1:
            if loosely_compare(q, q_sp_tokens):
                error_cnt = 0
            else:
                error_cnt += 1
                if error_cnt > 4:
                    print("query: ", q)
                    print("q_sp_tokens: ", q_sp_tokens)
                    raise ValueError()

        q_sb_tokens: list[list[str]] = lmap(tokenizer2.tokenize, q_sp_tokens)

        sb_st, sb_ed = translate_sp_idx_to_sb_idx(q_sb_tokens, st, ed)
        segment1 = RangePartitionedSegment(lflatten(q_sb_tokens), sb_st, sb_ed)

        segment2 = tokenizer2.tokenize(d1)
        pair_data = PairData(q, d1, "1", str(data_idx))
        data_idx += 1
        yield Segment1PartitionedPair(segment1, segment2, pair_data)

        segment2 = tokenizer2.tokenize(d2)
        pair_data = PairData(q, d2, "0", str(data_idx))
        data_idx += 1
        yield Segment1PartitionedPair(segment1, segment2, pair_data)


def loosely_compare(query: str, q_tokens):
    n_match = 0
    s1 = "".join(query.split()).lower()
    s2 = "".join(q_tokens).lower()

    for c1, c2 in zip(s1, s2):
        if c1 == c2:
            n_match += 1
            if n_match > 30:
                break
        else:
            print("s1: ", s1)
            print("s2: ", s2)
            return False
    return True