import collections
from misc_lib import average
import path
import os
import pickle





RawResult = collections.namedtuple("RawResult",
                                   ["unique_ids", "losses"])


def main():
    info_d = {}
    for job_id in range(5):
        p = os.path.join(path.data_path, "tlm", "pred", "info_d_{}.pickle".format(job_id))
        d = pickle.load(open(p, "rb"))
        info_d.update(d)

    p = os.path.join(path.data_path, "tlm", "pred", "tlm1.pickle")
    pred = pickle.load(open(p, "rb"))

    p_l = list([list() for i in range(5)])

    tf_id_set = set()

    for e in pred:
        tf_id = info_d[e.unique_ids]
        if tf_id not in tf_id_set:
            tf_id_set.add(tf_id)
            loss = e.losses
            print(tf_id, e.unique_ids, loss)
            j = e.unique_ids % 10
            p_l[j].append(loss)

    for i in range(5):
        print("Type : {} : {}".format(i, average(p_l[i])))





if __name__ == "__main__":
    main()
