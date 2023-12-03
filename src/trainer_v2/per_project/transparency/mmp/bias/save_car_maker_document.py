from dataset_specific.msmarco.passage.doc_indexing.index_path_helper import get_bm25_bt2_resource_path_helper
from dataset_specific.msmarco.passage.runner.bt2_inv_index import iter_bert_tokenized_merged
from misc_lib import TELI
from typing import List, Iterable, Callable, Dict, Tuple, Set
from cpath import output_path
from misc_lib import path_join



def main():
    collection_size = 8841823
    itr = iter_bert_tokenized_merged()
    itr = TELI(itr, collection_size)
    corpus_tokenized: Iterable[Tuple[str, List[str]]] = itr
    term_list_path = path_join(output_path, "mmp", "bias", "car_maker_list.txt")
    term_list = [line.lower().strip() for line in open(term_list_path, "r")]
    target_terms = set(term_list)

    save_path = path_join(output_path, "mmp", "bias", "car_maker_texts.tsv")

    f = open(save_path, "w")
    def save_item(doc_id, tokens):
        s = " ".join(tokens)
        f.write(f"{doc_id}\t{s}\n")

    for doc_id, word_tokens in itr:
        do_save = False
        for t in set(word_tokens):
            if t in target_terms:
                do_save = True
                break

        if do_save:
            save_item(doc_id, word_tokens)



if __name__ == "__main__":
    main()
