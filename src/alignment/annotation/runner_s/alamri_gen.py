import itertools
import os
import shutil
from typing import Iterator

from alignment.annotation.html_gen import get_get_html_fn
from contradiction.medical_claims.annotation_1.load_data import load_dev_pairs
from cpath import output_path, src_path
from dataset_specific.mnli.mnli_reader import NLIPairData
from misc_lib import exist_or_mkdir


def enum_alamri_pairs(split):
    if split == "dev":
        grouped_pairs = load_dev_pairs()
    elif split == "test":
        grouped_pairs = NotImplemented
    else:
        raise ValueError

    g_idx = 0
    for i, sent_pairs in grouped_pairs:
        for inner_idx, (text1, text2) in enumerate(sent_pairs):
            yield NLIPairData(text1, text2, "contradiction", str(g_idx))
            g_idx += 1


def align_annotation_gen(split, n_item, save_dir):
    itr: Iterator[NLIPairData] = enum_alamri_pairs(split)
    items = itertools.islice(itr, n_item)
    get_html = get_get_html_fn()
    for item in items:
        html = get_html(item)
        save_path = os.path.join(save_dir, "{}.html".format(item.data_id))
        with open(save_path, "w", errors="ignore") as f:
            f.write(html)


def main():
    split = 'dev'
    n_item = 100
    dataset = "alamri"
    save_name = "{}_{}_A".format(dataset, split)
    save_dir = os.path.join(output_path, "align_nli", "html_gen", save_name)
    exist_or_mkdir(save_dir)
    app_js = os.path.join(src_path, "html", "token_annotation", "app.js")
    dst = os.path.join(save_dir, "app.js")
    shutil.copyfile(app_js, dst)
    align_annotation_gen(split, n_item, save_dir)


if __name__ == "__main__":
    main()