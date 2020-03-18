import sys

from arg.perspectives.clueweb_helper import load_doc


def work(doc_id):
    doc = load_doc(doc_id)
    print(" ".join(doc))


if __name__ == "__main__":
    work(sys.argv[1])