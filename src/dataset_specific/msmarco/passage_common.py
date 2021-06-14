from typing import Tuple, Iterator

from cpath import at_data_dir


def enum_passage_corpus() -> Iterator[Tuple[str, str]]:
    msmarco_passage_corpus_path = at_data_dir("msmarco", "collection.tsv")
    with open(msmarco_passage_corpus_path, 'r', encoding='utf8') as f:
        for line in f:
            passage_id, text = line.split("\t")
            yield passage_id, text
