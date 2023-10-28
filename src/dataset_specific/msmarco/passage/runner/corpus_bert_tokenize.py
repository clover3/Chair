from transformers import AutoTokenizer

from transformers import AutoTokenizer

from cpath import output_path
from dataset_specific.msmarco.passage_common import enum_passage_corpus
from misc_lib import path_join, TimeEstimator


def main():
    collection_size = 8841823
    save_path = path_join(output_path, "msmarco", "passage_bert_tokenized.tsv")
    f = open(save_path, "w")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ticker = TimeEstimator(collection_size)
    itr = enum_passage_corpus()
    for doc_id, text in itr:
        tokens = tokenizer.tokenize(text)
        ticker.tick()
        tokens_str = " ".join(tokens)
        f.write(f"{doc_id}\t{tokens_str}\n")



if __name__ == "__main__":
    main()