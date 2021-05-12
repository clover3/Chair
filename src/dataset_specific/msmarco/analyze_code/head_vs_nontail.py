from list_lib import lmap
from misc_lib import group_by, get_first, get_second, get_third, two_digit_float
from tab_print import tab_print
from trec.qrel_parse import load_qrels_structured
from trec.trec_parse import load_ranked_list_grouped
import sys
from typing import List, Iterable, Callable, Dict, Tuple, Set, NamedTuple

from trec.types import TrecRankedListEntry, QRelsDict


def main():
    non_combine_ranked_list: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(sys.argv[1])
    judgement: QRelsDict = load_qrels_structured(sys.argv[2])

    # TODO : Find the case where scoring first segment would get better score
    # TODO : Check if the FP, TP document has score in low range or high range
    # TODO : Check if FP

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

    for query_id, entries in non_combine_ranked_list.items():
        new_e_list = []
        for e in entries:
            doc_id, passage_idx = parse_doc_name(e.doc_id)
            new_e = Passage(doc_id, e.score, passage_idx)
            new_e_list.append(new_e)
        grouped: Dict[str, List[Passage]] = group_by(new_e_list, get_first)
        get_passage_idx = get_third
        get_score = get_second

        grouped_entries = []
        doc_id_to_entry = {}
        score_by_head: List[Tuple[str, float]] = []
        for doc_id, scored_passages in grouped.items():
            scored_passages.sort(key=get_passage_idx)
            first_seg_score = scored_passages[0].score
            score_by_head.append((doc_id, first_seg_score))

            scored_passages.sort(key=Passage.get_score, reverse=True)
            max_score = scored_passages[0].score
            max_passage_idx = scored_passages[0].passage_idx
            num_passage = len(scored_passages)
            e = Entry(doc_id, max_score, max_passage_idx, num_passage, scored_passages)
            scored_passages.sort(key=get_passage_idx)
            grouped_entries.append(e)
            doc_id_to_entry[doc_id] = e

        rel_d = judgement[query_id]
        def is_relevant(doc_id):
            return doc_id in rel_d and rel_d[doc_id]

        score_by_head.sort(key=get_second, reverse=True)
        rel_rank_by_head = -1

        rel_rank = 0
        rel_doc_id = ""
        for rank, (doc_id, score) in enumerate(score_by_head):
            if is_relevant(doc_id):
                rel_rank_by_head = rank
                rel_doc_id = doc_id

        grouped_entries.sort(key=lambda x: x.max_score, reverse=True)
        rel_rank_by_max = -1
        for rank, e in enumerate(grouped_entries):
            if is_relevant(e.doc_id):
                rel_rank_by_max = rank

        def get_passage_score_str(passages: List[Passage]):
            passage_scores: List[float] = lmap(Passage.get_score, passages)
            scores_str = " ".join(map(two_digit_float, passage_scores))
            return scores_str

        if rel_rank_by_head < rel_rank_by_max:
            print()
            print("< Relevant document >")
            print("Rank by head", rel_rank_by_head)
            print("Rank by max", rel_rank_by_max)
            rel_entry = doc_id_to_entry[rel_doc_id]
            print(get_passage_score_str(rel_entry.passages))
            print("Num passages", rel_entry.num_passage)

            for rank, entry in enumerate(grouped_entries):
                if len(entry.passages) > 1 and entry.doc_id != rel_doc_id and rank < rel_rank_by_max:
                    print("< False positive document >")
                    print("Rank by max:", rank)
                    print("doc_id", entry.doc_id)
                    print("Num passages", entry.num_passage)
                    print("max_score", entry.max_score)
                    print("Passages scores: ", get_passage_score_str(entry.passages))
                    break


if __name__ == "__main__":
    main()

