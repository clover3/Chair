from typing import Tuple, Iterator

from cpath import at_data_dir


def enum_passage_corpus() -> Iterator[Tuple[str, str]]:
    msmarco_passage_corpus_path = at_data_dir("msmarco", "collection.tsv")
    yield from enum_two_column_tsv(msmarco_passage_corpus_path)


def enum_two_column_tsv(tsv_path):
    with open(tsv_path, 'r', encoding='utf8') as f:
        for line in f:
            data_id, text = line.split("\t")
            yield data_id, text
