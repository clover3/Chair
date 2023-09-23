import os
import pathlib
from typing import List, Iterable, Callable, Dict, Tuple, Set
from cpath import output_path, data_path
from misc_lib import path_join
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader


DictSS = Dict[str, Dict]
DictSI = Dict[str, Dict[str, int]]


def load_beir_dataset(dataset_name, split)\
        -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
    #### Download scifact.zip dataset and unzip the dataset
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset_name)
    beir_root = path_join(data_path, "beir")
    out_dir = path_join(beir_root, "datasets")
    save_path = util.download_and_unzip(url, out_dir)

    #### Provide the data path where scifact has been downloaded and unzipped to the data loader
    # data folder would contain these files:
    # (1) scifact/corpus.jsonl  (format: jsonlines)
    # (2) scifact/queries.jsonl (format: jsonlines)
    # (3) scifact/qrels/test.tsv (format: tsv ("\t"))
    corpus, queries, qrels = GenericDataLoader(save_path).load(split=split)
    return corpus, queries, qrels


beir_dataset_list_not_large = [
    "hotpotqa",
    "dbpedia-entity",
    "nq",
    "webis-touche2020",
    "scidocs",
    "trec-covid-beir",
    "trec-covid",
    "trec-covid-v2",
    "fiqa",
    "quora",
    "arguana",
    "scifact",
    "nfcorpus",
    "vihealthqa"
]


def main():
    corpus, queries, qrels = load_beir_dataset("scifact", "test")
    print(f"Corpus has {len(corpus)} entries")
    print(type(corpus))
    return NotImplemented


if __name__ == "__main__":
    main()