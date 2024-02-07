import pickle

from dataset_specific.msmarco.passage.passage_resource_loader import load_qrel, enum_queries, load_msmarco_collection
from dataset_specific.msmarco.passage.tokenize_helper import iter_tokenized_corpus
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
    save_path = path_join(output_path, "mmp", "passage_lucene_k", "all.tsv")
    itr = iter_tokenized_corpus(save_path)
    corpus = dict(itr)

    print("Done")
    counter = Counter()
    qrel = load_qrel("train")
    print("qrel has {} items".format(len(qrel)))

    ticker = TimeEstimator(700000)
    for qid, query in enum_queries("train"):
        ticker.tick()
        try:
            rel_doc_id = []
            for doc_id, val in qrel[qid].items():
                if val > 0:
                    rel_doc_id.append(doc_id)

            q_tokens = tokenize_fn(query)

            for doc_id in rel_doc_id:
                pos_text = corpus[doc_id]
                d_tokens = tokenize_fn(pos_text)
                for q_term in q_tokens:
                    for d_term in d_tokens:
                        counter[(q_term, d_term)] += 1
        except KeyError as e:
            pass

    save_dir = path_join(output_path, "mmp", "lucene_krovetz")
    save_path = path_join(save_dir, "rel_tf.pkl")
    pickle.dump(counter, open(save_path, "wb"))


def main():
    work_fn()


if __name__ == "__main__":
    main()
