import os
import random

from dataset_specific.msmarco.passage.grouped_reader import get_train_query_grouped_dict_10K
from job_manager.job_runner_with_server import JobRunnerF
from misc_lib import path_join
from trainer_v2.chair_logging import c_log


def filter_with_keyword(group_no, keyword="when"):
    c_log.info("Loading from group %s", group_no)
    d = get_train_query_grouped_dict_10K(group_no)
    c_log.info("Done")

    for query_id, entries in d.items():
        try:
            _qid, _pid, query, _text = entries[0]
            if keyword in query:
                pass
            else:
                continue

            yield from entries
        except ValueError as e:
            print("Entries:", len(entries))
            print(e)


def work_fn(job_no):
    keyword = "when"
    st = job_no * 10
    ed = st + 10
    save_dir = path_join("output", "msmarco", "passage", "when_full")
    save_path = os.path.join(save_dir, str(job_no))
    f = open(save_path, "w")
    for group_no in range(st, ed):
        for item in filter_with_keyword(group_no, keyword):
            f.write("\t".join(item) + "\n")


def main():
    working_dir = path_join("output", "msmarco", "passage")
    runner = JobRunnerF(working_dir, 11, "when_full", work_fn)
    runner.start()


if __name__ == "__main__":
    main()