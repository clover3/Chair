import pickle
from collections import Counter

from cpath import output_path
from dataset_specific.msmarco.passage.tokenize_helper import iter_tokenized_corpus
from misc_lib import path_join, TimeEstimator
from trainer_v2.chair_logging import c_log


def main():
    collection_size = 8841823
    save_path = path_join(output_path, "mmp", "passage_lucene_k", "all.tsv")
    itr = iter_tokenized_corpus(save_path)
    c_log.info("Building inverted index")
    ticker = TimeEstimator(collection_size)
    all_count = Counter()
    for doc_id, word_tokens in itr:
        count = Counter(word_tokens)
        all_count.update(count)
        ticker.tick()

    save_path = path_join(output_path, "mmp", "lucene_krovetz", "tf.pkl")
    pickle.dump(all_count, open(save_path, "wb"))



if __name__ == "__main__":
    main()
