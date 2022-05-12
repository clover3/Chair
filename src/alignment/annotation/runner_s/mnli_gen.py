import itertools
import os
import shutil
from typing import Iterator

from alignment.annotation.html_gen import get_get_html_fn
from cpath import output_path, src_path
from dataset_specific.mnli.mnli_reader import MNLIReader, NLIPairData
from misc_lib import exist_or_mkdir


def mnli_gen(split, n_item, save_dir):
    mnli_reader = MNLIReader()
    itr: Iterator[NLIPairData] = mnli_reader.load_split(split)
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
    dataset = "mnli"
    save_name = "{}_{}_A".format(dataset, split)
    save_dir = os.path.join(output_path, "align_nli", "html_gen", save_name)
    exist_or_mkdir(save_dir)
    app_js = os.path.join(src_path, "html", "token_annotation", "app.js")
    dst = os.path.join(save_dir, "app.js")
    shutil.copyfile(app_js, dst)

    mnli_gen(split, n_item, save_dir)


if __name__ == "__main__":
    main()