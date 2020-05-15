####
import csv
import os
from pathlib import Path
from typing import Iterator, List
from zipfile import ZipFile

from arg.counter_arg import header
from arg.counter_arg.header import ArguDataPoint, Passage, ArguDataID
from base_type import FilePath
from cpath import data_path, pjoin

corpus_root_dir: FilePath = pjoin(data_path, "arguana-counterargs-corpus")
#####

crawled_webpages = pjoin(corpus_root_dir, "01-crawled-webpages")
extracted_arguments = pjoin(corpus_root_dir, "02-extracted-arguments")
pair_best_counter = pjoin(corpus_root_dir, "03-pairs-best-counter-task")


def extract_zip_file_at(zip_file_path, dir_path):
    # Create a ZipFile Object and load sample.zip in it
    with ZipFile(zip_file_path, 'r') as zipObj:
        # Extract all the contents of zip file in different directory
        zipObj.extractall(dir_path)


def load_tsv_or_from_zip(dir_path, file_name) -> Iterator:
    file_path = pjoin(dir_path, file_name)
    if not os.path.exists(file_path):
        print("extracting from zip...")
        zip_file_path = file_path + ".zip"
        extract_zip_file_at(zip_file_path, dir_path)

    f = open(file_path, "r", encoding="utf-8", errors="ignore")
    reader = csv.reader(f, delimiter='\t')

    for idx, row in enumerate(reader):
        yield row


def load_label(split, topic) -> Iterator:
    split_dir = pjoin(pair_best_counter, split)

    topic_dir = pjoin(split_dir, topic)

    file_list = [
        "01-debate-opposing-counters.tsv",
        "02-debate-counters.tsv",
        "03-debate-opposing-arguments.tsv",
        "04-debate-arguments.tsv",
        "05-theme-counters.tsv",
        "06-theme-arguments.tsv"
    ]

    return load_tsv_or_from_zip(topic_dir, file_list[4])


def load_labeled_data_per_topic(split, topic) -> List[ArguDataPoint]:
    def load_passage_from_rel_path(rel_path: ArguDataID) -> Passage:
        tokens = rel_path.id.split("/")
        tokens = [t if t != "con" else "_con" for t in tokens]
        conv_rel_path = Path(os.sep.join(tokens))
        file_path = os.path.join(extracted_arguments, conv_rel_path)
        text = open(file_path, "r", encoding="utf-8").read()
        return Passage(text=text, id=rel_path)

    r = []
    for row in load_label(split, topic):
        annotations = row[2:]
        if annotations[0] == "true":
            pass
        else:
            continue

        text1 = load_passage_from_rel_path(ArguDataID.from_name(row[0]))
        text2 = load_passage_from_rel_path(ArguDataID.from_name(row[1]))

        e = ArguDataPoint(text1, text2, annotations)
        r.append(e)
    return r


def load_labeled_data(split) -> List[ArguDataPoint]:
    r = []
    for topic in header.topics:
        itr = load_labeled_data_per_topic(split, topic)
        for item in itr:
            r.append(item)
    return r
