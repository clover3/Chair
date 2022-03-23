from typing import List, Dict, Tuple

from arg.counter_arg_retrieval.build_dataset.judgments import Judgement, JudgementEx
from arg.counter_arg_retrieval.build_dataset.passage_scoring.split_passages import PassageRange
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText
from trec.types import DocID, TrecRankedListEntry


def convert_to_judgment_entries(passages: Dict[DocID, Tuple[SegmentwiseTokenizedText, List[PassageRange]]],
                                pq1: Dict[str, List[TrecRankedListEntry]]) -> List[Judgement]:
    all_judgements = []
    for qid, entries in pq1.items():
        for e in entries:
            doc_id, passage_idx = e.doc_id.split("_")
            passage_idx = int(passage_idx)
            swtt, passage_ranges = passages[doc_id]
            st, ed = passage_ranges[passage_idx]
            judgement = Judgement(qid, doc_id, passage_idx, st, ed)
            all_judgements.append(judgement)
    return all_judgements


def convert_to_judgment_ex_entries(passages: Dict[DocID, Tuple[SegmentwiseTokenizedText, List[PassageRange]]],
                                pq1: Dict[str, List[TrecRankedListEntry]]) -> List[JudgementEx]:
    all_judgements = []
    for qid, entries in pq1.items():
        for e in entries:
            doc_id, passage_idx = e.doc_id.split("_")
            passage_idx = int(passage_idx)
            swtt, passage_ranges = passages[doc_id]
            st, ed = passage_ranges[passage_idx]
            judgement = JudgementEx(qid, doc_id, passage_idx, st, ed, swtt)
            all_judgements.append(judgement)
    return all_judgements

