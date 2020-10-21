import os

from arg.perspectives.relevance_and_doc_value.write_html_todo import load_qk
from cpath import output_path
from datastore.interface import load_multiple
from datastore.table_names import RawCluewebDoc
from misc_lib import exist_or_mkdir


def load_doc_ids2():
    f = open(os.path.join(output_path, "doc_ids_with_effect.txt"), "r")
    for line in f:
        try:
            doc_id = line.split()[1]
            yield doc_id
        except IndexError:
            pass


def load_doc_ids():
    doc_ids = set()
    for query, k_list in load_qk():
        for k in k_list:
            doc_ids.add(k.doc_id)
    return doc_ids


def main():
    doc_ids = list(set(load_doc_ids()))
    print("num docs", len(doc_ids))
    save_dir = os.path.join(output_path, "pc_docs_html")

    k = 0
    step = 1000
    while k < len(doc_ids):
        print(k, k+step)
        cur_doc_ids = doc_ids[k:k+step]
        docs = load_multiple(RawCluewebDoc, cur_doc_ids, True)
        exist_or_mkdir(save_dir)
        for doc_id in cur_doc_ids:
            try:
                doc_html = docs[doc_id]
                save_path = os.path.join(save_dir, doc_id + ".html")
                open(save_path, "w").write(doc_html)
            except KeyError:
                pass
        k += step




if __name__ == "__main__":
    main()