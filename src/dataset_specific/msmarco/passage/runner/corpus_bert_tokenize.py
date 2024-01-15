from transformers import AutoTokenizer
from dataset_specific.msmarco.passage_common import enum_passage_corpus

from cpath import output_path
from dataset_specific.msmarco.passage.tokenize_helper import corpus_tokenize
from misc_lib import path_join


def main():
    collection_size = 8841823
    save_path = path_join(output_path, "msmarco", "passage_bert_tokenized.tsv")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenize_fn = tokenizer.tokenize
    itr = enum_passage_corpus()

    corpus_tokenize(itr, tokenize_fn, save_path, collection_size)

if __name__ == "__main__":
    main()