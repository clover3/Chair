from clueweb.clueweb_galago_db import DocGetter
from galagos.interface import send_doc_queries, get_doc
from sydney_clueweb.clue_path import get_first_disk


def basic_test():
    queries = [{
        "number": "test1",
        "text": "#combine(controversial donald trump president)"
    }]
    print(send_doc_queries(get_first_disk(), 10, queries))
    get_doc(get_first_disk(), "clueweb12-0011wb-80-12217")


def doc_getter_test():
    doc_getter = DocGetter()
    doc_id = "clueweb12-0001wb-96-00013"
    r = doc_getter.get_doc_tf(doc_id)
    print(r)

    ###


if __name__ == "__main__":
    basic_test()
