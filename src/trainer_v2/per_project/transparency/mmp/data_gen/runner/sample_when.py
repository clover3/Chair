import os

from job_manager.job_runner_with_server import JobRunnerF
from misc_lib import path_join
from trainer_v2.per_project.transparency.mmp.data_enum import enum_pos_neg_sample


def iterate_pairs():
    itr = enum_pos_neg_sample(range(0, 109))
    cnt = 0
    when_cnt = 0
    for item in itr:
        cnt += 1
        query_text, pos_text, neg_text = item
        if 'when' in query_text.lower():
            when_cnt += 1
            yield item
            if when_cnt % 100 == 0:
                print("{}/{} = {}".format(when_cnt, cnt, when_cnt / cnt))


def work_fn(job_no):
    st = job_no * 10
    ed = st + 10
    save_dir = path_join("output", "msmarco", "passage", "when")
    save_path = os.path.join(save_dir, str(job_no))
    f = open(save_path, "w")
    itr = enum_pos_neg_sample(range(st, ed))
    for item in itr:
        query_text, pos_text, neg_text = item
        if 'when' in query_text.lower():
            f.write("\t".join(item) + "\n")



def main():
    working_dir = path_join("output", "msmarco", "passage")
    runner = JobRunnerF(working_dir, 11, "when", work_fn)
    runner.start()


if __name__ == "__main__":
    main()