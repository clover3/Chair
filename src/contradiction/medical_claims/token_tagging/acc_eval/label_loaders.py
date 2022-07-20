import os
from typing import List

from cache import load_list_from_jsonl
from cpath import output_path


class SentTokenLabel:
    def __init__(self, qid, labels):
        self.qid = qid
        self.labels = labels

    def to_json(self):
        return {
            'qid': self.qid,
            'labels': self.labels,
        }

    @classmethod
    def from_json(cls, j):
        qid = j['qid']
        labels = j['labels']
        assert type(qid) == str
        for item in labels:
            assert type(item) == int
        return SentTokenLabel(qid, labels)


def load_sbl_label(split) -> List[SentTokenLabel]:
    if split not in ["val", "test"]:
        print("Only \'val\' or \'test\' is expected")
    file_path = os.path.join(output_path, "alamri_annotation1", "label", "sbl.qrel.{}".format(split))
    return load_sent_token_label(file_path)


def load_sent_token_label(file_path) -> List[SentTokenLabel]:
    return load_list_from_jsonl(file_path, SentTokenLabel.from_json)

