import json
from typing import List, Tuple

from trec.types import QRelsFlat, QRelsDict, QRelsSubtopic, DocID, QueryID


def load_qrels_flat_per_query(path) -> QRelsFlat:
    # 101001 0 clueweb12-0001wb-40-32733 0
    q_group = dict()
    for line in open(path, "r"):
        q_id, _, doc_id, score = line.split()
        score = int(score)
        if q_id not in q_group:
            q_group[q_id] = list()

        q_group[q_id].append((doc_id, int(score)))
    return q_group


def load_json_qrel(path) -> QRelsFlat:
    j_d = json.load(open(path, "r"))
    out_j_d = {}
    for qid, items in j_d.items():
        entries = []
        for doc_id, score in items.items():
            score = 1 if score else 0
            entries.append((doc_id, score))
        out_j_d[qid] = entries
    return out_j_d


def load_qrels_flat_per_query_0_1_only(path) -> QRelsFlat:
    if path.endswith(".json"):
        return load_json_qrel(path)
    # 101001 0 clueweb12-0001wb-40-32733 0
    q_group = dict()
    for line in open(path, "r"):
        q_id, _, doc_id, score = line.split()
        score = int(score)
        if q_id not in q_group:
            q_group[q_id] = list()

        score = 1 if score else 0
        q_group[q_id].append((doc_id, score))
    return q_group


def load_qrels_all_flat(path) -> List[Tuple[QueryID, DocID, int]]:
    output = []
    for line in open(path, "r"):
        q_id, _, doc_id, score = line.split()
        output.append((str(q_id), str(doc_id), int(score)))
    return output



def load_qrels_with_subtopic(path) -> QRelsSubtopic:
    # 101001 2 clueweb12-0001wb-40-32733 0
    q_group = dict()
    for line in open(path, "r"):
        q_id, subtopic, doc_id, score = line.split()
        score = int(score)
        if q_id not in q_group:
            q_group[q_id] = list()

        q_group[q_id].append((doc_id, subtopic, int(score)))
    return q_group


def load_qrels_structured(path) -> QRelsDict:
    f = open(path, "r")
    q_dict = {}
    for line in f:
        q_id, _, doc_id, score = line.split()
        if q_id not in q_dict:
            q_dict[q_id] = {}

        q_dict[q_id][doc_id] = int(score)
    return q_dict