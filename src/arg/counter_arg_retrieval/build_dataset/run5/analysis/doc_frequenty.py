from collections import defaultdict
from typing import NamedTuple

from misc_lib import group_by
from trec.types import TrecRelevanceJudgementEntry, load_qrel_as_entries


class _Entry(NamedTuple):
    doc_id: str
    passage_idx: int
    score: int

    @classmethod
    def from_trec_entry(cls, entry: TrecRelevanceJudgementEntry):
        tokens = entry.doc_id.split("_")
        doc_id = "_".join(tokens[:-1])
        passage_idx = int(tokens[-1])
        return _Entry(doc_id, passage_idx, entry.relevance)


def main():
    qrel_save_path = "C:\\work\\Code\\Chair\\output\\ca_building\\qrel\\0522_done.txt"
    qrel = load_qrel_as_entries(qrel_save_path)

    n_relevant = 0
    n_multi_relevance = 0
    n_passage = 0
    n_non_single_group = 0
    n_relevant_and_non_single_group = 0
    n_exist_other_relevance = 0

    for qid, entries in qrel.items():
        entries_g = list(map(_Entry.from_trec_entry, entries))
        entries_g_relevant = list(filter(lambda x: x.score > 0, entries_g))
        grouped = group_by(entries_g, lambda x:  x.doc_id)
        grouped_relevant = group_by(entries_g_relevant, lambda x:  x.doc_id)

        for e in entries_g:
            n_passage += 1

            if e.score > 0:
                n_relevant += 1

            if len(grouped[e.doc_id]) > 1:
                n_non_single_group += 1

            if e.score > 0 and len(grouped[e.doc_id]) > 1:
                n_relevant_and_non_single_group += 1
            doc_id = e.doc_id
            n_group_relevant = len(grouped_relevant[doc_id]) if doc_id in grouped_relevant else 0
            if e.score > 0:
                n_group_relevant_other_than_me = n_group_relevant -1
            else:
                n_group_relevant_other_than_me = n_group_relevant

            if n_group_relevant_other_than_me > 0:
                n_exist_other_relevance += 1

            # if e.score > 0:
            #     if n_group_relevant > 1:
            #         n_exist_other_relevance += 1
            # else:
            #     if n_group_relevant > 0:
            #         n_exist_other_relevance += 1

            if e.score > 0 and n_group_relevant > 1:
                n_multi_relevance += 1
        #
        #
        # for doc_id, items in grouped_relevant.items():
        #     g_size = len(items)
        #     n_relevant += g_size
        #     if g_size > 1:
        #         n_multi_relevance += g_size
        #     if len(grouped[doc_id]) > 1:
        #         n_relevant_and_non_single_group += g_size
        #
        # for doc_id, items in grouped.items():
        #     g_size = len(items)
        #     n_passage += g_size
        #     if g_size > 1:
        #         n_non_single_group += g_size
        #     relevant = grouped_relevant[doc_id] if doc_id in grouped_relevant else []
        #     if len(relevant) == 1:
        #         exist_other_relevance = g_size-1
        #     elif len(relevant) > 1:
        #         exist_other_relevance = (g_size - len(relevant)) + (len(relevant)-1)
        #     else:
        #         exist_other_relevance = 0
        #     n_exist_other_relevance += exist_other_relevance

    print("n_relevant", n_relevant)
    print("n_relevant_and_non_single_group", n_multi_relevance)
    print("n_passage", n_passage)
    print("n_non_single_group", n_non_single_group)
    print("P(R(P_i) | exist P_i, P_j in D s.t R(P_j)) = {}".format(n_multi_relevance / n_exist_other_relevance))
    print("P(non_single_group) = {}".format(n_non_single_group / n_passage))
    print("P(non_single_group|R) = {}".format(n_relevant_and_non_single_group / n_relevant))
    print("P(R|non_single_group) = {}".format(n_relevant_and_non_single_group / n_non_single_group))
    print("P(R) = {}".format(n_relevant / n_passage))

    # P(Rel) vs P(Rel| other passage Rel)


def global_stats():
    qrel_save_path = "C:\\work\\Code\\Chair\\output\\ca_building\\qrel\\0522_done.txt"
    qrel = load_qrel_as_entries(qrel_save_path)

    claim_grouped = defaultdict(list)
    for qid, entries in qrel.items():
        entries_g = list(map(_Entry.from_trec_entry, entries))
        cid, _pid = qid.split("_")
        claim_grouped[cid].extend(entries_g)


    n_relevant = 0
    n_relevant_and_non_single_group = 0
    n_multi_relevance = 0
    n_passage = 0
    n_non_single_group = 0
    n_exist_other_relevance = 0
    for cid, entries_g in claim_grouped.items():
        entries_g_relevant = list(filter(lambda x: x.score > 0, entries_g))
        grouped = group_by(entries_g, lambda x:  x.doc_id)
        grouped_relevant = group_by(entries_g_relevant, lambda x:  x.doc_id)

        for e in entries_g:
            n_passage += 1

            if e.score > 0:
                n_relevant += 1

            if len(grouped[e.doc_id]) > 1:
                n_non_single_group += 1

            if e.score > 0 and len(grouped[e.doc_id]) > 1:
                n_relevant_and_non_single_group += 1
            doc_id = e.doc_id
            n_group_relevant = len(grouped_relevant[doc_id]) if doc_id in grouped_relevant else 0

            if e.score > 0:
                if n_group_relevant > 1:
                    n_exist_other_relevance += 1
            else:
                if n_group_relevant > 0:
                    n_exist_other_relevance += 1

            if e.score > 0 and n_group_relevant > 1:
                n_multi_relevance += 1

    print("n_relevant", n_relevant)
    print("n_relevant_and_non_single_group", n_relevant_and_non_single_group)
    print("n_passage", n_passage)
    print("n_non_single_group", n_non_single_group)
    print("P(R(P_i) | exist, P_i, P_j, s.t R(P_j)) = {}".format(n_multi_relevance / n_exist_other_relevance))
    print("P(non_single_group) = {}".format(n_non_single_group / n_passage))
    print("P(non_single_group|R) = {}".format(n_relevant_and_non_single_group / n_relevant))
    print("P(R|non_single_group) = {}".format(n_relevant_and_non_single_group / n_non_single_group))
    print("P(R) = {}".format(n_relevant / n_passage))

    # P(Rel) vs P(Rel| other passage Rel)


if __name__ == "__main__":
    main()
