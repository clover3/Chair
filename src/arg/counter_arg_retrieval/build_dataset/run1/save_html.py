import csv
import json
import os

from cpath import output_path
from misc_lib import exist_or_mkdir


def main():
    save_path = os.path.join(output_path, "ca_building", "run1", "mturk_todo.csv")
    reader = csv.reader(open(save_path, "r"), delimiter=',')
    doc_ids = []
    for row in reader:
        try:
            doc_id = row[2]
            doc_ids.append(doc_id)
        except IndexError as e:
            print(e)

    save_path = os.path.join(output_path, "ca_building", "run1", "docs.jsonl")
    docs_d = {}
    for line_no, line in enumerate(open(save_path, "r", newline="\n")):
        try:
            j = json.loads(line, strict=False)
            docs_d[j['id']] = j['content']
        except json.decoder.JSONDecodeError:
            print(line)
            print("json.decoder.JSONDecodeError", line_no)

    save_dir = os.path.join(output_path, "ca_building", "run1", "html")
    exist_or_mkdir(save_dir)
    for doc_id in doc_ids:
        content = docs_d[doc_id]
        doc_save_path = os.path.join(save_dir, "{}.html".format(doc_id))
        open(doc_save_path, "w", encoding="utf-8").write(content)


if __name__ == "__main__":
    main()
