from collections import Counter

from adhoc.kn_tokenizer import KrovetzNLTKTokenizer
from cache import save_to_pickle
from cpath import output_path
import os

from dataset_specific.msmarco.passage.passage_resource_loader import tsv_iter
from misc_lib import path_join


#  Count term frequency in "when" corpus
def enum_when_corpus():
    for job_no in range(11):
        save_dir = path_join("output", "msmarco", "passage", "when")
        save_path = path_join(save_dir, str(job_no))
        yield from tsv_iter(save_path)


def main():
    tokenizer = KrovetzNLTKTokenizer()
    tf_for = {
        'q': Counter(),
        'd+': Counter(),
        'd-': Counter(),
    }

    for query, doc_pos, doc_neg in enum_when_corpus():
        todo = {
            'q': query,
            'd+': doc_pos,
            'd-': doc_neg,
        }
        for role, text in todo.items():
            tokens = tokenizer.tokenize_stem(text)
            for t in tokens:
                tf_for[role][t] += 1

    save_to_pickle(tf_for, "tf_for")
    for role, tf in tf_for.items():
        print("")
        print(role)
        for term, cnt in tf.most_common(200):
            print(f"{term}\t{cnt}")


if __name__ == "__main__":
    main()
##