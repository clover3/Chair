import os

from cpath import output_path
from datastore.interface import load_multiple
from datastore.table_names import RawCluewebDoc
from misc_lib import exist_or_mkdir


def load_doc_ids():
    f = open(os.path.join(output_path, "doc_ids_with_effect.txt"), "r")
    for line in f:
        try:
            doc_id = line.split()[1]
            yield doc_id
        except IndexError:
            pass


def main():
    doc_ids = set(load_doc_ids())
    print(doc_ids)
    docs = load_multiple(RawCluewebDoc, doc_ids, True)

    save_dir = os.path.join(output_path, "pc_docs_html")
    exist_or_mkdir(save_dir)
    for doc_id in doc_ids:
        try:
            doc_html = docs[doc_id]
            save_path = os.path.join(save_dir, doc_id + ".html")
            open(save_path, "w").write(doc_html)
        except KeyError:
            pass



if __name__ == "__main__":
    main()