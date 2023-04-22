from cache import load_from_pickle
from math import log

def main():
    tf, rel_tf = load_from_pickle("when0_rel_tf")
    n_rel_doc = 231
    n_all_doc = 231 * 1000
    n_neg_doc = n_all_doc - n_rel_doc
    entries = []
    for key in rel_tf:
        n_pos = rel_tf[key]
        n_all = tf[key]
        n_neg = n_all - n_pos

        pos_log = log(n_pos / n_rel_doc)
        neg_log = log(n_neg / n_neg_doc)
        log_odd = pos_log - neg_log


        entries.append((key, n_pos, n_neg, log_odd))

    entries.sort(key=lambda x: x[3], reverse=True)
    for e in entries:
        print(e)


if __name__ == "__main__":
    main()