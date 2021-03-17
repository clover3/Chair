import sys

from tab_print import print_table
from trec.qrel_parse import load_qrels_structured
from trec.trec_parse import load_ranked_list_grouped, write_trec_ranked_list_entry
from trec.types import QRelsDict, TrecRankedListEntry


def main():
    rlg_proposed_tfidf = load_ranked_list_grouped(sys.argv[1])
    rlg_proposed_bm25 = load_ranked_list_grouped(sys.argv[2])
    rlg_bert_tfidf = load_ranked_list_grouped(sys.argv[3])
    qrel: QRelsDict = load_qrels_structured(sys.argv[4])

    # TODO
    #  Q1 ) Is the set of document different?
    #  Q2 ) Say 2 (BM25) is better than 1 (tf-idf),
    #    1. X=0, in 2 not in 1
    #    2. X=1, in 2 not in 1
    #    3. X=0, in 1 not in 2   -> FP prediction that BERT(baseline) misses
    #    4. X=1, in 1 not in 2
    cnt = 0
    for q in rlg_proposed_tfidf:
        entries1 = rlg_proposed_tfidf[q]
        entries2 = rlg_proposed_bm25[q]
        entries3 = rlg_bert_tfidf[q]
        e3_d = {e.doc_id: e for e in entries3}

        def get_doc_set(entries):
            return set(map(TrecRankedListEntry.get_doc_id, entries))
        docs1 = get_doc_set(entries1)
        docs2 = get_doc_set(entries2)

        d = qrel[q]
        rows = [[q]]
        rows.append(['doc_id', 'label', 'in_bm25', '1_rank', '1_score', '3_rank', '3_score'])
        for e in entries1:
            label = d[e.doc_id] if e.doc_id in d else 0
            #if e.doc_id not in docs2:
            if True:
                # Case 3
                predict_binary = e.rank < 20
                try:
                    e3 = e3_d[e.doc_id]
                    row = [e.doc_id, label, e.doc_id in docs2, e.rank, e.score, e3.rank, e3.score]
                    rows.append(row)
                except KeyError:
                    assert cnt ==0
                    cnt += 1

        if len(rows) > 2:
            print_table(rows)

def main2():
    rlg_proposed_tfidf = load_ranked_list_grouped(sys.argv[1])
    rlg_proposed_bm25 = load_ranked_list_grouped(sys.argv[2])
    rlg_bert_tfidf = load_ranked_list_grouped(sys.argv[3])
    qrel: QRelsDict = load_qrels_structured(sys.argv[4])

    flat_etr1 = []
    flat_etr3 = []
    for q in rlg_proposed_tfidf:
        entries1 = rlg_proposed_tfidf[q]
        entries2 = rlg_proposed_bm25[q]
        entries3 = rlg_bert_tfidf[q]
        def get_doc_set(entries):
            return set(map(TrecRankedListEntry.get_doc_id, entries))

        docs2 = get_doc_set(entries2)

        d = qrel[q]

        def reform(entries):
            es = list([e for e in entries if e.doc_id not in docs2])

            new_entries = []
            for idx, e in enumerate(es):
                new_entries.append(TrecRankedListEntry(e.query_id, e.doc_id, idx, e.score, e.run_name))
            return new_entries

        etr1 = reform(entries1)
        flat_etr1.extend(etr1)
        etr3 = reform(entries3)
        flat_etr3.extend(etr3)

    write_trec_ranked_list_entry(flat_etr1, "bm25equi_proposed.txt")
    write_trec_ranked_list_entry(flat_etr3, "bm25equi_bert.txt")


if __name__ == "__main__":
    main2()

