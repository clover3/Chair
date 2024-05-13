import os
from collections import OrderedDict

def main():
    qid_set = OrderedDict()
    for job_no in range(120):
        save_path = f"output/msmarco/passage/mmp_train_split_ranked_list/{job_no}.txt"
        if not os.path.exists(save_path):
            continue

        f = open(save_path, "r")
        n_added = 0
        for line in f:
            qid = line.split(" ")[0]
            if qid not in qid_set:
                qid_set[qid] = 1
                n_added += 1

            if n_added > 10:
                break

    sampled_qids = f"output/msmarco/passage/train_sample_qids.txt"
    f = open(sampled_qids, "w")
    for qid in qid_set:
        f.write("{}\n".format(qid))


if __name__ == "__main__":
    main()
