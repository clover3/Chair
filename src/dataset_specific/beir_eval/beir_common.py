import csv
import os
import pathlib
from typing import List, Iterable, Callable, Dict, Tuple, Set
from cpath import output_path, data_path
from iter_util import load_jsonl
from misc_lib import path_join
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader


DictSS = Dict[str, Dict]
DictSI = Dict[str, Dict[str, int]]


beir_dataset_list_A = [
    # It excludes 4 non-public datasets and MSMARCO
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
    # cqadupstack
    # climate-fever
]


beir_gb_dataset_list = [
    # It excludes 4 non-public datasets and MSMARCO
    "hotpotqa",
    "dbpedia-entity",
    "nq",
]

# nq: 4 hours
# hotpotqa: 17 hours, (1110 min expected)
# quora: 4 hours


# webis-touche2020 scidocs trec-covid fiqa quora arguana scifact nfcorpus vihealthqa
beir_mb_dataset_list = [
    # size < 311MB
    "webis-touche2020",
    "scidocs",
    "trec-covid",
    "fiqa",
    "quora",
    "arguana",
    "scifact",
    "nfcorpus",
    "vihealthqa"
]

avdl_luk = {
    "hotpotqa": 30,
    "dbpedia-entity": 32,
    "nq": 53,
    "webis-touche2020": 193,
    "scidocs": 119,
    "trec-covid-beir": 111,
    "trec-covid": 111,
    "trec-covid-v2": 147,
    "fiqa": 91,
    "quora": 8,
    "arguana": 108,
    "scifact": 151,
    "nfcorpus":165,
    "vihealthqa": 110
}

# Given information, we will construct a JSON dictionary where each key is the dataset name and its value is the ctf.

ctf_luk = {
    "arguana": 944123,
    "dbpedia-entity": 152205479,
    "fiqa": 5288635,
    "hotpotqa": 158180692,
    "nfcorpus": 601950,
    "nq": 144050891,
    "quora": 4390852,
    "scidocs": 3065828,
    "scifact": 784591,
    "trec-covid": 19060122,
    "trec-covid-beir": 19060122,
    "trec-covid-v2": 19060115,
    "vihealthqa": 1088953,
    "webis-touche2020": 74066724,
}


def load_beir_dataset(dataset_name, split)\
        -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
    #### Download scifact.zip dataset and unzip the dataset
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset_name)
    beir_root = path_join(data_path, "beir")
    out_dir = path_join(beir_root, "datasets")
    downloaded_path = path_join(out_dir, dataset_name)
    if not os.path.exists(downloaded_path):
        print(f"Downloading BEIR {dataset_name}")
        save_path = util.download_and_unzip(url, out_dir)
        downloaded_path = save_path
    #### Provide the data path where scifact has been downloaded and unzipped to the data loader
    # data folder would contain these files:
    # (1) scifact/corpus.jsonl  (format: jsonlines)
    # (2) scifact/queries.jsonl (format: jsonlines)
    # (3) scifact/qrels/test.tsv (format: tsv ("\t"))
    corpus, queries, qrels = GenericDataLoader(downloaded_path).load(split=split)
    return corpus, queries, qrels


def get_beir_queries(dataset_name) -> List[Tuple[str, str]]:
    beir_root = path_join(data_path, "beir")
    file_path = path_join(beir_root, "datasets", dataset_name, "queries.jsonl")
    return [(q["_id"], q['text']) for q in load_jsonl(file_path)]


def parse_beir_qrels(qrels_file) -> Dict[str, Dict[str, int]]:
    reader = csv.reader(open(qrels_file, encoding="utf-8"),
                        delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    next(reader)
    qrels = {}
    for id, row in enumerate(reader):
        query_id, corpus_id, score = row[0], row[1], int(row[2])

        if query_id not in qrels:
            qrels[query_id] = {corpus_id: score}
        else:
            qrels[query_id][corpus_id] = score
    return qrels


def load_beir_qrels(dataset_name, split) -> Dict[str, Dict[str, int]]:
    beir_root = path_join(data_path, "beir")
    file_path = path_join(beir_root, "datasets", dataset_name, "qrels", f"{split}.tsv")
    return parse_beir_qrels(file_path)


def load_beir_queries_and_qrels(dataset_name, split) -> Tuple:
    queries = get_beir_queries(dataset_name)
    qrels = load_beir_qrels(dataset_name, split)
    queries = [(qid, _text) for qid, _text in queries if qid in qrels]
    return queries, qrels


def main():
    # corpus, queries, qrels = load_beir_dataset("scifact", "test")
    # print(f"Corpus has {len(corpus)} entries")
    # print(type(corpus))
    for dataset_name in beir_dataset_list_A:
        print(dataset_name)
        queries = get_beir_queries(dataset_name)
        for q in queries:
            print(q["_id"], q['text'])
            break



if __name__ == "__main__":
    main()