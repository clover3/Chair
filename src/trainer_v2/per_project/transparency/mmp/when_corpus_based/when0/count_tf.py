from collections import Counter

from cache import save_to_pickle
from dataset_specific.msmarco.passage.passage_resource_loader import load_qrel
from trainer_v2.per_project.transparency.mmp.when_corpus_based.when0.data_enum import load_when0_corpus
from trainer_v2.per_project.transparency.mmp.when_corpus_based.when_bm25t import load_align_weights


def main():
    min_tf = 10
    terms = set()
    for t in load_align_weights():
        if t.n_appear >= min_tf:
            terms.add(t.word)

    print("{} terms selected".format(len(terms)))
    qrel = load_qrel("train")
    tf = Counter()
    rel_tf = Counter()
    itr = load_when0_corpus()
    for qid, pid, qtf, ptf in itr:
        if pid in qrel[qid] and qrel[qid][pid]:
            rel = True
        else:
            rel = False
        for t in ptf:
            if t in terms:
                tf[t] += ptf[t]

                if rel:
                    rel_tf[t] += ptf[t]

    output = tf, rel_tf

    for t in tf.keys():
        print(t, tf[t], rel_tf[t])

    save_to_pickle(output, "when0_rel_tf")


if __name__ == "__main__":
    main()