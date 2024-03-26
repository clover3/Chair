import csv
from cpath import output_path
from misc_lib import path_join

import torch
from torch.utils.data import DataLoader, TensorDataset
import unittest
from transformers import AutoTokenizer
from ptorch.splade_tree.datasets.pep_dataloaders import PartitionEncoder
import itertools


def iter_brown_corpus_sents(n=100):
    # Create a simple dataset for demonstration purposes
    import nltk
    # Download the Brown Corpus
    # nltk.download('brown')
    # Import the Brown Corpus
    from nltk.corpus import brown
    # Access the text of the Brown Corpus
    brown_text = brown.sents()  # This gives you the raw text of the corpus
    for t in itertools.islice(brown_text, n):
        yield " ".join(t)


class TestPartitionEncoder(unittest.TestCase):
    def setUp(self):

        max_seq_length = 256
        tokenizer_type = "bert-base-uncased"
        self.batch_size = 10
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
        self.encoder = PartitionEncoder(self.tokenizer, max_seq_length)

    def test_q_partitions(self):
        n = 1
        sentences = list(iter_brown_corpus_sents(n * 2))
        entries = []

        for i in range(n):
            query = sentences[i * 2]
            doc = sentences[i * 2 + 1]
            q_rep = self.tokenizer(query)
            d_rep = self.tokenizer(doc)
            q_indices = self.encoder.partition_query(q_rep[0])
            d_indices: list[int] = self.encoder.partition_doc(d_rep[0])
            encoded = self.encoder.combine(q_rep[0], d_rep[0], q_indices, d_indices)
            entries.append(encoded)

        doc_encoded = self.encoder.as_batch_arrays(entries)
        save_path1 = path_join(output_path, "pep_dev", "linear_inputs.csv")
        save_path2 = path_join(output_path, "pep_dev", "mat_inputs.csv")
        f1 = csv.writer(open(save_path1, "w", newline='', encoding="utf-8"))
        f2 = csv.writer(open(save_path2, "w", newline='', encoding="utf-8"))

        for i in range(n):
            for role in ["left", "right"]:
                ret = doc_encoded
                input_ids = ret[f'{role}_input_ids'][i]
                token_type_ids = ret[f'{role}_token_type_ids'][i]
                doc_cls_indices = ret[f'{role}_doc_cls_indices'][i]

                f1.writerow([f"{i}_{role}"])
                f1.writerow(input_ids)
                f1.writerow(token_type_ids)
                f1.writerow(doc_cls_indices)

                f2.writerow([f"{i}_{role}"])
                attn_list = ret[f'{role}_attention_mask'][i]
                for j in range(len(attn_list)):
                    f2.writerow(attn_list[j])


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
