import os

from cpath import output_path
from trec.trec_parse import load_ranked_list_grouped


def main():
    rlg_path = os.path.join(output_path, "ca_building", "run5", "q_res.txt")
    rlg = load_ranked_list_grouped(rlg_path)
    qids = list(rlg.keys())
    qids.sort()

    qid_path = os.path.join(output_path, "ca_building", "run5", "qids.txt")

    f = open(qid_path, "w")
    for qid in qids:
        f.write("{}\n".format(qid))


if __name__ == "__main__":
    main()