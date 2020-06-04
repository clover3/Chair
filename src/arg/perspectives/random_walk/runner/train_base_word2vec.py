
import os
import time

import gensim.models

from base_type import FilePath
from cache import load_pickle_from
from cpath import output_path
from misc_lib import get_dir_files


def load_corpus():
    dir_path = FilePath("/mnt/nfs/work3/youngwookim/data/bert_tf/clueweb12_13B_word_tokens/")

    corpus = []

    cnt = 0
    for file_path in get_dir_files(dir_path):
        tokens_list = load_pickle_from(file_path)
        corpus.extend(tokens_list)
        if cnt > 50:
            break
        cnt += 1
    return corpus


def main():
    print("Loading corpus..")
    corpus = load_corpus()
    print("Total of {} sentences".format(len(corpus)))
    print("Building model..")
    begin = time.time()
    model = gensim.models.Word2Vec(sentences=corpus, min_count=30, workers=16)
    end = time.time()

    print("Elapsed : ", end-begin)
    save_path = os.path.join(output_path, "word2vec_clueweb12_13B")
    model.save(save_path)


if __name__ == "__main__":
    main()
