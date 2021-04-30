from typing import List, Iterable, Callable, Dict, Tuple, Set, NamedTuple, Iterator
import csv
from galagos.types import Query


class Doc(NamedTuple):
    url: str
    doc_id: str
    title: str
    content: str


def read_triplet(file_path) -> Iterator[Tuple[Query, Doc, Doc]]:
    # for row in csv.reader(open(file_path, "r"), delimiter="\t"):
    fail_cnt = 0
    suc_cnt = 0
    for line in open(file_path, "r"):
        try:
            qid, query_text, doc_id_1, url1, title1, content1, doc_id_2, url2, title2, content2 = line.split("\t")
            query = Query(qid, query_text)
            doc1 = Doc(url1, doc_id_1, title1, content1)
            doc2 = Doc(url2, doc_id_2, title2, content2)
            yield query, doc1, doc2
        except ValueError:
            fail_cnt += 1


def main():
    print("Main")
    read_triplet("/mnt/nfs/work3/youngwookim/data/msmarco/triples.tsv")


if __name__ == "__main__":
    main()
