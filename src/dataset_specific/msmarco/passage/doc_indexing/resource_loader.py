from cpath import output_path
from table_lib import tsv_iter
from misc_lib import path_join
from typing import List, Iterable, Tuple


def enum_msmarco_passage_tokenized() -> Iterable[Tuple[str, List[str]]]:
    work_path = path_join(output_path, "msmarco", "msmarco_passage_tokenize")

    num_item = 9

    for i in range(num_item):
        file_path = path_join(work_path, str(i))
        for row in tsv_iter(file_path):
            print(row)
            doc_id, doc = row
            yield doc_id, doc.split()



def enum_msmarco_passage_tokenized() -> Iterable[Tuple[str, List[str]]]:
    work_path = path_join(output_path, "msmarco", "msmarco_passage_tokenize", "all")
    for row in tsv_iter(work_path):
        doc_id, doc = row
        yield doc_id, doc.split()
