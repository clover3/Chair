import sys
from collections import Counter
from typing import List, Dict, Tuple, NamedTuple

from misc_lib import group_by, get_first, get_third
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def main():
    non_combine_ranked_list: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(sys.argv[1])

    def parse_doc_name(doc_name):
        tokens = doc_name.split("_")
        doc_id = "_".join(tokens[:-1])
        passage_idx = int(tokens[-1])
        return doc_id, passage_idx

    class Passage(NamedTuple):
        doc_id: str
        score: float
        passage_idx: int

        def get_doc_id(self):
            return self.doc_id

        def get_score(self):
            return self.score

    class Entry(NamedTuple):
        doc_id: str
        max_score: float
        max_passage_idx: int
        num_passage: int
        passages: List[Passage]

    counter = Counter()

    qids = list(non_combine_ranked_list.keys())
    qids.sort()
    seen_seg_id = set()
    for qid in qids:
        entries = non_combine_ranked_list[qid]
        new_e_list = []
        for e in entries:
            doc_id, passage_idx = parse_doc_name(e.doc_id)

            key = qid, doc_id, passage_idx
            assert key not in seen_seg_id
            seen_seg_id.add(key)
            new_e = Passage(doc_id, e.score, passage_idx)
            new_e_list.append(new_e)
        grouped: Dict[str, List[Passage]] = group_by(new_e_list, get_first)
        get_passage_idx = get_third
        score_by_head: List[Tuple[str, float]] = []
        for doc_id, scored_passages in grouped.items():
            scored_passages.sort(key=get_passage_idx)
            first_seg_score = scored_passages[0].score
            score_by_head.append((doc_id, first_seg_score))

            scored_passages.sort(key=Passage.get_score, reverse=True)
            max_score = scored_passages[0].score
            max_passage_idx = scored_passages[0].passage_idx

            if len(scored_passages) == 1:
                continue
            if max_passage_idx == 0:
                counter["first"] += 1
            elif max_passage_idx == len(scored_passages)-1:
                counter["last"] += 1
            else:
                counter["middle"] += 1
            counter["all"] += 1

    for key, value in counter.items():
        print(key, value)


if __name__ == "__main__":
    main()

