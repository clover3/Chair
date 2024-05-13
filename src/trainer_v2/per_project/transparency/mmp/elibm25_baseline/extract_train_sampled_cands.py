import os.path
from collections import OrderedDict

from cpath import output_path, data_path
from misc_lib import path_join
from trainer_v2.per_project.transparency.mmp.train_all_inf.score_from_gz import tsv_iter_from_gz


def main():
    qid_path = "output/msmarco/passage/train_sample_qids.txt"
    train_sample_qids = [t.strip() for t in open(qid_path, "r").readlines()]
    train_sample_qids = train_sample_qids[:1000]
    train_sample_qids = set(train_sample_qids)
    save_path = path_join(data_path, "msmarco", "passage", "train_sample", "corpus.tsv")
    f = open(save_path, "w")

    found = OrderedDict()
    for job_no in range(120):
        quad_tsv_path = path_join(data_path, "msmarco", "passage", "group_sorted_10K_gz", f"{job_no}.gz")
        if not os.path.exists(quad_tsv_path):
            continue

        non_match_count = 0
        for row in tsv_iter_from_gz(quad_tsv_path):
            if row[0] in train_sample_qids:
                found[row[0]] = 1
                f.write("\t".join(row) + "\n")
            else:
                non_match_count += 1

            if non_match_count > 5000:
                break

    print(f"{len(found)} queries found")


if __name__ == "__main__":
    main()
