import pickle

from dataset_specific.msmarco.passage.passage_resource_loader import load_qrel, enum_queries, load_msmarco_collection
from trainer_v2.per_project.transparency.mmp.data_enum import enum_pos_neg_sample
from collections import Counter
from cpath import output_path
from misc_lib import path_join, TimeEstimator


def work_fn():
    print("use lucene_krovetz")
    from pyserini.analysis import Analyzer, get_lucene_analyzer
    analyzer = Analyzer(get_lucene_analyzer(stemmer='krovetz'))
    tokenize_fn = analyzer.analyze
    print("loading collections")
    corpus = dict(load_msmarco_collection())
    print("Done")
    counter = Counter()
    qrel = load_qrel("train")
    print("qrel has {} items")



    ticker = TimeEstimator(400000)
    for qid, query in enum_queries("train"):
        ticker.tick()
        tokens = tokenize_fn(query)
        if "car" in tokens:
            rel_doc_id = []
            try:
                for doc_id, val in qrel[qid].items():
                    if val > 0:
                        rel_doc_id.append(doc_id)

                # print(qid, rel_doc_id, query)

                for doc_id in rel_doc_id:
                    pos_text = corpus[doc_id]
                    for d_token in tokenize_fn(pos_text):
                        counter[d_token] += 1
            except KeyError as e:
                pass

    save_dir = path_join(output_path, "mmp", "car_rel_tf")
    save_path = path_join(save_dir, "all.pkl")
    pickle.dump(counter, open(save_path, "wb"))


def main():
    work_fn()


if __name__ == "__main__":
    main()
