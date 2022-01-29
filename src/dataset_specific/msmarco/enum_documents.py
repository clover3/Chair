from typing import Iterable

from dataset_specific.msmarco.common import at_working_dir, MSMarcoDoc


def enum_documents() -> Iterable[MSMarcoDoc]:
    doc_f = open(at_working_dir("msmarco-docs.tsv"), encoding="utf8")
    for line in doc_f:
        docid, url, title, body = line.split("\t")
        yield MSMarcoDoc(docid, url, title, body)


if __name__ == "__main__":
    enum_documents()