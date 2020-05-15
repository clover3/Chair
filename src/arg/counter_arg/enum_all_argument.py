import os
from typing import Iterable

from arg.counter_arg import header
from arg.counter_arg.data_loader import extracted_arguments
from arg.counter_arg.header import Passage, ArguDataID
from cpath import pjoin
from misc_lib import get_dir_files


def get_rel_path(full_path, head_path):
    st = len(head_path)
    return full_path[st+1:]


def enum_all_argument(split) -> Iterable[Passage]:
    assert split in header.splits
    all_topic_dir = pjoin(extracted_arguments, split)

    for topic in header.topics:
        per_topic_dir = pjoin(all_topic_dir, topic)
        for maybe_dir_obj in os.scandir(per_topic_dir):
            if not maybe_dir_obj.is_dir():
                continue
            dir_path = maybe_dir_obj.path

            con_dir = pjoin(dir_path, "_con")
            pro_dir = pjoin(dir_path, "pro")

            def load_files_in_dir(target_dir_path):
                assert os.path.basename(target_dir_path) in ["_con", "pro"]
                for file_path in get_dir_files(target_dir_path):
                    content = open(file_path, "r", encoding='utf-8').read()
                    rel_path = get_rel_path(file_path, extracted_arguments)
                    yield Passage(content, ArguDataID.from_rel_path(rel_path))

            for item in load_files_in_dir(con_dir):
                yield item
            for item in load_files_in_dir(pro_dir):
                yield item


if __name__ == "__main__":
    cnt = 0
    for item in enum_all_argument("training"):
        print(item.id)
        cnt += 1
    print(cnt)
