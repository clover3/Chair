from trec.qrel_parse import load_qrels_all_flat


def count_true_rate(qrel_path):
    entries = load_qrels_all_flat(qrel_path)
    true_cnt = 0
    neg_cnt = 0
    for e in entries:
        _, _, score = e
        if score > 0:
            true_cnt += 1
        else:
            neg_cnt += 1

    return true_cnt, neg_cnt