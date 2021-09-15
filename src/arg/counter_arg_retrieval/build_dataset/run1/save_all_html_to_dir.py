import os

from cpath import output_path
from galagos.jsonl_util import load_jsonl
from misc_lib import exist_or_mkdir, TimeEstimator


def main():
    save_path = os.path.join(output_path, "ca_building", "run1", "docs.jsonl")
    docs_d = load_jsonl(save_path)

    save_dir = os.path.join(output_path, "ca_building", "run1", "html")
    exist_or_mkdir(save_dir)
    ticker = TimeEstimator(len(docs_d))
    for doc_id, content in docs_d.items():
        doc_save_path = os.path.join(save_dir, "{}.html".format(doc_id))
        open(doc_save_path, "w", encoding="utf-8").write(content)
        ticker.tick()


if __name__ == "__main__":
    main()
