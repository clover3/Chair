import json
import json
import os
from collections import defaultdict
from typing import List, Iterable, Dict, Tuple
from typing import NamedTuple

from list_lib import lfilter
from misc_lib import get_dir_dir, get_dir_files

root_dir = "C:\work\Data\MPQA\database.mpqa.2.0"


class MPQARawDoc(NamedTuple):
    doc_id: str
    content: str


class MPQAAnnLine(NamedTuple):
    id: int
    span: Tuple[int,int]
    data_type: str
    ann_type: str
    attributes: str


class Sentence(NamedTuple):
    id: int
    span: Tuple[int, int]
    subjectivity_annot: List[str]
    ann_lines: List[MPQAAnnLine]


class MPQADocSubjectiveInfo(NamedTuple):
    doc_id: str
    content: str
    sentences: List[Sentence]

    def get_text(self, s: Sentence):
        st, ed = s.span
        return self.content[st:ed]


def load_all_docs() -> List[MPQARawDoc]:
    docs = []
    doc_dir_path = os.path.join(root_dir, "docs")
    for parent_dir in get_dir_dir(doc_dir_path):
        parent_name = os.path.basename(parent_dir)
        for doc_leaf_path in get_dir_files(parent_dir):
            file_name = os.path.basename(doc_leaf_path)
            doc_id = parent_name + "/" + file_name
            try:
                content = open(doc_leaf_path, "r", encoding="utf-8").read()
                docs.append(MPQARawDoc(doc_id, content))
            except UnicodeDecodeError:
                print(doc_leaf_path)
                raise
    return docs


def read_mqpa_anns(file_path) -> List[MPQAAnnLine]:
    ann_list: List[MPQAAnnLine] = []
    for line in open(file_path, "r"):
        if not line.strip():
            continue

        if line[0] == "#":
            continue
        row = line.strip().split("\t")
        id = row[0]
        span_st, span_ed = row[1].split(",")
        data_type = row[2]
        ann_type = row[3]
        if len(row) > 4:
            attributes = row[4]
        else:
            attributes = ""
        ann = MPQAAnnLine(int(id), (int(span_st), int(span_ed)), data_type, ann_type, attributes)
        ann_list.append(ann)

    return ann_list


def load_all_annotations() -> List[Tuple[str, List[MPQAAnnLine]]]:
    doc_dir_path = os.path.join(root_dir, "man_anns")
    for parent_dir in get_dir_dir(doc_dir_path):
        parent_name = os.path.basename(parent_dir)
        for doc_leaf_path in get_dir_dir(parent_dir):
            doc_leaf_name = os.path.basename(doc_leaf_path)
            doc_id = parent_name + "/" + doc_leaf_name
            ann_set_list = []
            for ann_file_path in get_dir_files(doc_leaf_path):
                ann_file_type = os.path.basename(ann_file_path)
                assert ann_file_type in ["gateman.mpqa.lre.2.0", "gatesentences.mpqa.2.0", "answer.mpqa.2.0"]
                lines = read_mqpa_anns(ann_file_path)
                ann_set_list.extend(lines)
                yield doc_id, ann_set_list


EXPRESSIVE_SUBJECTIVITY = "GATE_expressive-subjectivity"
DIRECT_SUBJECTIVITY = "GATE_direct-subjective"
OBJECTIVE = "GATE_objective-speech-event"

num_error = 0


def combine_subjectivity_annotation(doc: MPQARawDoc, ann_list: List[MPQAAnnLine]) -> MPQADocSubjectiveInfo:
    def is_sentence_annot(ann: MPQAAnnLine) -> bool:
        return ann.ann_type == "GATE_sentence"
    # identify sentences
    sentences = lfilter(is_sentence_annot, ann_list)
    sentences.sort(key=lambda s:s.span[0])
    if not sentences:
        print(ann_list)
    assert sentences

    def is_it_about_subjective(ann: MPQAAnnLine) -> bool:
        return ann.ann_type in [EXPRESSIVE_SUBJECTIVITY, DIRECT_SUBJECTIVITY]

    # filter subjectivity related ones
    ann_about_subjectivity = lfilter(is_it_about_subjective, ann_list)

    def find_sentence(span) -> int:
        st, ed = span
        for s in sentences:
            st_s, ed_s = s.span
            if st_s <= st and ed <= ed_s:
                return s.id

        raise KeyError()

    # Match sentence with annotation
    global num_error

    s_list_to_ann_list: Dict[int, List] = defaultdict(list)
    for annot in ann_about_subjectivity:
        try:
            if annot.span == (0, 0):
                continue
            sentence_id = find_sentence(annot.span)
            s_list_to_ann_list[sentence_id].append(annot)
        except KeyError:
            num_error += 1


    annot_sent_list = []
    for raw_sent in sentences:
        ann_list = s_list_to_ann_list[raw_sent.id]
        tags = list([a.ann_type for a in ann_list])
        annot_sent = Sentence(raw_sent.id, raw_sent.span, tags, ann_list)
        annot_sent_list.append(annot_sent)

    return MPQADocSubjectiveInfo(doc.doc_id, doc.content, annot_sent_list)


def get_all_subjectivity_data() -> Iterable[MPQADocSubjectiveInfo]:
    anns: Dict[str, List[MPQAAnnLine]] = dict(load_all_annotations())

    docs = load_all_docs()

    for doc in docs:
        if doc.doc_id in anns:
            ann_lines: List[MPQAAnnLine] = anns[doc.doc_id]
            doc_and_info: MPQADocSubjectiveInfo = combine_subjectivity_annotation(doc, ann_lines)
            yield doc_and_info


splits = ["train", "dev", 'test']


def build_split():
    l = list(get_all_subjectivity_data())
    train_len = int(len(l) * 0.8)
    val_len = int(len(l) * 0.1)
    train = l[:train_len]
    dev = l[train_len:train_len+val_len]
    test = l[train_len+val_len:]

    split_d = {}
    for doc in train:
        split_d[doc.doc_id] = 'train'
    for doc in dev:
        split_d[doc.doc_id] = 'dev'
    for doc in test:
        split_d[doc.doc_id] = 'test'

    save_path = os.path.join(root_dir, "split.json")
    json.dump(split_d, open(save_path, "w"))


def load_for_split(split) -> Iterable[MPQADocSubjectiveInfo]:
    l = list(get_all_subjectivity_data())
    splid_d = load_split_dict()

    for doc in l:
        if splid_d[doc.doc_id] == split:
            yield doc


def load_split_dict():
    save_path = os.path.join(root_dir, "split.json")
    splid_d = json.load(open(save_path, "r"))
    return splid_d


def main():
    l = list(get_all_subjectivity_data())
    pos_cnt = 0
    sent_cnt = 0
    for doc in l:
        for s in doc.sentences:
            sent_cnt += 1
            if s.subjectivity_annot:
                pos_cnt += 1
                print("S", doc.get_text(s))
            else:
                print("O", doc.get_text(s))
    print("num error:", num_error)
    print("num docs", len(l))
    print("Num_pos", pos_cnt)
    print("sent cnt", sent_cnt)


if __name__ == "__main__":
    build_split()

