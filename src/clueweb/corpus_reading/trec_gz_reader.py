import gzip
import sys
from typing import List, Iterable, Tuple


def iter_docs(file_path) -> Iterable[Tuple[str, List[str]]]:
    f = gzip.open(file_path, "rt")
    IN_DOC = 1
    OUT_DOC = 2
    state = OUT_DOC
    doc_id = None
    cur_doc = []
    for idx, line in enumerate(f):

        ##
        if state == OUT_DOC:
            if line.startswith("<DOC>"):
                cur_doc.append(line)
                state = IN_DOC
        elif state == IN_DOC:
            cur_doc.append(line)
            if line.startswith("</DOC>"):
                yield doc_id, cur_doc
                doc_id = None
                cur_doc = []
                state = OUT_DOC

            if doc_id is None:
                if line.startswith("<DOCNO>"):
                    st = len("<DOCNO>")
                    line_strip = line.strip()
                    ed = len(line_strip) - len("</DOCNO>")
                    # assert line_strip[ed:] == "</DOCNO>"
                    assert doc_id is None
                    doc_id = line_strip[st:ed]

        else:
            assert False

            ##


def read_add(file_path):
    f = gzip.open(file_path, "rt")
    data = []
    for idx, line in enumerate(f):
        if idx % 3 == 0:
            data.append(line)
    return data


def name_enum():
    names = set()
    for doc in iter_docs(sys.argv[1]):
        doc_id, _ = doc
        corpus_name, name1, name2, name3 = doc_id.split("-")
        key = name1, name2
        if not key in names:
            print(key)
            names.add(key)

        ## break


if __name__ == "__main__":
    # name_enum()
    read_add(sys.argv[1])