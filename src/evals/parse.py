from evals.types import QRelsFlat, QRelsDict


def load_qrels_flat(path) -> QRelsFlat:
    # 101001 0 clueweb12-0001wb-40-32733 0
    q_group = dict()
    for line in open(path, "r"):
        q_id, _, doc_id, score = line.split()
        score = int(score)
        if q_id not in q_group:
            q_group[q_id] = list()

        q_group[q_id].append((doc_id, int(score)))
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