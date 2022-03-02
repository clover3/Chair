import csv

from arg.counter_arg_retrieval.build_dataset.run5.path_helper import load_qids, get_premise_query_path
from arg.perspectives.evaluate import perspective_getter


def save_tsv(output, save_path):
    tsv_writer = csv.writer(open(save_path, "w", newline=""), delimiter="\t")
    tsv_writer.writerows(output)


def main():
    def get_p_text(qid):
        cid_s, pid_s = qid.split("_")
        perspective_text = perspective_getter(int(pid_s))
        return perspective_text

    qids = load_qids()
    output = []
    for qid in qids:
        output.append((qid, get_p_text(qid)))

    save_path = get_premise_query_path()
    save_tsv(output, save_path)


if __name__ == "__main__":
    main()