import sys

from clueweb.add_doc_list_to_db import read_doc_list
from clueweb.clueweb_galago_db import get_docs_in_db


def main():
    doc_list = read_doc_list(sys.argv[1])
    table_name = sys.argv[2]
    docs_in_db = get_docs_in_db(table_name)
    for doc in doc_list:
        if doc not in docs_in_db:
            print(doc)


if __name__ == "__main__":
    main()