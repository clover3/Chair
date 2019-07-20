
def load_galago_judgement(path):
# Sample Format : 475287 Q0 LA053190-0016_1274 1 15.07645119 galago
    q_group = dict()
    for line in open(path, "r"):
        q_id, _, doc_id, rank, score, _ = line.split()
        if q_id not in q_group:
            q_group[q_id] = list()
        q_group[q_id].append((doc_id, int(rank), float(score)))
    return q_group