from typing import NamedTuple, Tuple, List, Dict, Iterator

from data_generator.bert_input_splitter import split_p_h_with_input_ids
from tlm.qtype.content_functional_parsing.qid_to_content_tokens import QueryInfo
from tlm.qtype.qtype_instance import QTypeInstance


class TwoPieceQueryPart(NamedTuple):
    head: str
    tail: str

    @classmethod
    def from_tuple(cls, tuple: Tuple[str, str]):
        h, t = tuple
        return TwoPieceQueryPart(h, t)

    def __str__(self):
        return "{} [MASK] {}".format(self.head, self.tail)


def query_info_to_tuple(info: QueryInfo):
    head, tail = info.get_head_tail()
    return " ".join(head), " ".join(tail)


class Instance(NamedTuple):
    doc_tokens_ids: List
    query: str
    logits: float
    ft: TwoPieceQueryPart
    ct: str

    def get_query_rep(self):
        return "{} [{}] {}".format(self.ft.head, self.ct, self.ft.tail)


def enum_helper(qtype_entries: List[QTypeInstance], query_info_dict: Dict[str, QueryInfo]) -> Iterator[Instance]:
    seen = set()
    n_rel = 0
    n_non_rel = 0
    for e_idx, e in enumerate(qtype_entries):
        if e.label > 0.5:
            n_rel += 1
        else:
            if n_non_rel >= n_rel:
                continue
            n_non_rel += 1
        overlap_key = e.qid
        info: QueryInfo = query_info_dict[e.qid]
        if overlap_key in seen:
            continue

        ft_tuple = query_info_to_tuple(info)

        query, doc_tokens_ids = split_p_h_with_input_ids(e.de_input_ids, e.de_input_ids)
        yield Instance(doc_tokens_ids.tolist(), info.query, e.label,
                       TwoPieceQueryPart.from_tuple(ft_tuple), info.content_span)