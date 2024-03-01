from cpath import output_path
from dataset_specific.beir_eval.beir_common import load_beir_dataset
from misc_lib import path_join
from trec.trec_parse import load_ranked_list_grouped



def main():
    p = r"output/ranked_list/arguana_empty.txt"
    rlg = load_ranked_list_grouped(p)
    corpus, queries, qrels = load_beir_dataset("arguana", "test")

    qids = list(rlg.keys())
    qids.sort()

    qid = qids[0]
    print("Qid", qid)
    print()
    for idx, e in enumerate(rlg[qid]):
        # print(e)
        print()
        doc = corpus[e.doc_id]
        print("[{}]".format(idx+1))
        print(doc['text'])


if __name__ == "__main__":
    main()