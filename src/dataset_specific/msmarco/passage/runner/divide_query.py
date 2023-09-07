from collections import defaultdict, Counter

from cpath import output_path
from table_lib import tsv_iter
from misc_lib import path_join, TimeEstimator


def read_save_grouped(grouping_root, n_item, read_f):
    ticker = TimeEstimator(n_item, sample_size=1000)
    per_query_dict = defaultdict(list)
    per_query_count = Counter()
    f_dict = {}
    n_record = 0
    for line in read_f:
        # e = line.split("\t")
        idx1 = line.find("\t")
        idx2 = line.find("\t", idx1 + 1)
        qid = line[:idx1]
        pid = line[idx1 + 1: idx2]

        # qid, pid, _, _ = e
        n_record += 1
        group_id = str(int(int(qid) / 10000))
        per_query_count[qid] += 1

        if group_id not in f_dict:
            file_path = path_join(grouping_root, group_id)
            f = open(file_path, "w")
            f_dict[group_id] = f

        f = f_dict[group_id]

        f.write(line)
        ticker.tick()
        l = per_query_dict[qid]
        l.append(pid)
        if per_query_count[qid] == 1000:
            print(f"1000 records are found for {qid} (Read {n_record})")


def main():
    # 10K means that queries are grouped by dividing qid by 10K
    source_corpus_path = path_join("data", "msmarco", "top1000.train.txt")
    grouping_root = path_join("data", "msmarco", "passage", "grouped_10K")
    read_f = open(source_corpus_path, "r", encoding="utf-8", errors="ignore")
    n_item = 478002393

    read_save_grouped(grouping_root, n_item, read_f)


def main_val():
    source_corpus_path = path_join("data", "msmarco", "top1000.dev")
    grouping_root = path_join("data", "msmarco", "passage", "dev_grouped_10K")
    read_f = open(source_corpus_path, "r", encoding="utf-8", errors="ignore")
    n_item = 6668967

    read_save_grouped(grouping_root, n_item, read_f)


if __name__ == "__main__":
    main_val()