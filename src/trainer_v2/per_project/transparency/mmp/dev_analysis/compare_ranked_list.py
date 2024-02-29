from typing import List, Dict

from cpath import data_path
from misc_lib import path_join
from trec.qrel_parse import load_qrels_structured
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry
from dataset_specific.msmarco.passage.passage_resource_loader import load_msmarco_collection


def main():
    def load_ranked_list(name):
        print(f"loading from {name}")
        file_path = f"output/ranked_list/{name}.txt"
        l: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(file_path)
        return l

    corpus_d = dict(load_msmarco_collection())

    judgment_path = path_join(data_path, "msmarco", "qrels.dev.tsv")
    qrels = load_qrels_structured(judgment_path)
    rlg1 = load_ranked_list("rr_ce_msmarco_mini_lm_dev1K_A")
    rlg2 = load_ranked_list("rr_splade_dev1K_A")

    cnt = 0
    for qid in rlg1:
        rl1 = rlg1[qid]
        rl2 = rlg2[qid]
        try:
            label_d: dict[str, int] = qrels[qid]
            if len(label_d) > 1:
                print("More than one label for qid", qid, label_d)
                continue

            def split_ranked_list(rl):
                is_after_rel = False
                above_rel = []
                below_rel = []
                rel_loc = None
                for idx, e in enumerate(rl):
                    if e.doc_id in label_d and label_d[e.doc_id]:
                        rel_loc = idx
                        is_after_rel = True
                    else:
                        if is_after_rel:
                            below_rel.append(e)
                        else:
                            above_rel.append(e)
                if not rel_loc:
                    raise IndexError()
                return rl[rel_loc], above_rel, below_rel

            rel1, above_rel1, below_rel1 = split_ranked_list(rl1)
            rel2, above_rel2, below_rel2 = split_ranked_list(rl2)

            print("------------------")
            print("Qid", qid)
            print("Gold {} CE Rank={} Splade Rank={}".format(rel1.doc_id, rel1.rank, rel2.rank))
            print(corpus_d[rel1.doc_id])
            print()
            # Find document that r1 rank lower than rel_loc1,
            #                    r2 rank higher than rel_loc2,

            below_rel1_doc_ids = [e.doc_id for e in below_rel1]  # True negative
            above_rel2_doc_ids = [e.doc_id for e in above_rel2]  # False positive

            doc_ids = set(below_rel1_doc_ids).intersection(set(above_rel2_doc_ids))

            d_below_rel1 = {e.doc_id: e for e in below_rel1}
            d_above_rel2 = {e.doc_id: e for e in above_rel2}
            for doc_id in doc_ids:
                e1 = d_below_rel1[doc_id]
                e2 = d_above_rel2[doc_id]
                print("NR {} CE Rank={} Splade Rank={}".format(
                    doc_id, e1.rank, e2.rank))
                print(corpus_d[doc_id])
                print()

            for e in below_rel1:
                if e.doc_id in doc_ids:
                    assert e.score < rel1.score

            for e in above_rel2:
                if e.doc_id in doc_ids:
                    assert e.score > rel2.score

        except IndexError:
            pass
        cnt += 1
        if cnt > 20:
            break


if __name__ == "__main__":
    main()
