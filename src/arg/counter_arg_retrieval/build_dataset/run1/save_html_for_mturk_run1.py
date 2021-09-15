import csv
import os

from cpath import output_path
from galagos.jsonl_util import load_jsonl
from misc_lib import exist_or_mkdir


def load_doc_ids(save_path):
    reader = csv.reader(open(save_path, "r"), delimiter=',')
    doc_ids = []
    for row in reader:
        try:
            doc_id = row[2]
            doc_ids.append(doc_id)
        except IndexError as e:
            print(e)
    return doc_ids


def main():
    save_path = os.path.join(output_path, "ca_building", "run1", "mturk_todo.csv")
    doc_ids = load_doc_ids(save_path)

    save_path = os.path.join(output_path, "ca_building", "run1", "docs.jsonl")
    docs_d = load_jsonl(save_path)

    save_dir = os.path.join(output_path, "ca_building", "run1", "html")
    exist_or_mkdir(save_dir)
    for doc_id in doc_ids:
        content = docs_d[doc_id]
        doc_save_path = os.path.join(save_dir, "{}.html".format(doc_id))
        open(doc_save_path, "w", encoding="utf-8").write(content)


if __name__ == "__main__":
    main()
