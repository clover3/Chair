import tqdm
from transformers import AutoTokenizer
from ptorch.splade_tree.datasets.pep_dataloaders import PartitionEncoder
import os


def main():
    max_seq_length = 256
    tokenizer_type = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
    encoder = PartitionEncoder(tokenizer, max_seq_length)
    data_dir = "data/msmarco/splade_triplets"

    def data_iter():
        data_path = os.path.join(data_dir, "raw.tsv")
        with open(data_path) as reader:
            for i, line in enumerate(reader):
                if len(line) > 1:
                    yield line.split("\t")

    for i, e in enumerate(data_iter()):
        q, d_pos, d_neg, _s_pos, _s_neg = e
        print(q)
        q_rep = tokenizer(q)
        for d in [d_pos, d_neg]:
            d_rep = tokenizer(d)
            q_indices = encoder.partition_query(q_rep[0])
            d_indices: list[int] = encoder.partition_doc(d_rep[0])
            encoded = encoder.combine(q_rep[0], d_rep[0], q_indices, d_indices)
            left, right = encoded
            print(d)
            print(" ".join(tokenizer.convert_ids_to_tokens(left.input_ids)))
            print(" ".join(map(str, left.token_type_ids)))
            print(" ".join(map(str, left.doc_cls_indices)))
            print(" ".join(tokenizer.convert_ids_to_tokens(right.input_ids)))
            print(" ".join(map(str, right.token_type_ids)))
            print(" ".join(map(str, right.doc_cls_indices)))
            print()

        if i >= 10:
            break


if __name__ == "__main__":
    main()
