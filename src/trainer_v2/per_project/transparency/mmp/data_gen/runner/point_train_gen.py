
import os

from job_manager.job_runner_with_server import JobRunnerF
from misc_lib import path_join
from tf_util.record_writer_wrap import write_records_w_encode_fn
from trainer_v2.per_project.transparency.mmp.data_enum import enum_pos_neg_pointwise
from trainer_v2.per_project.transparency.mmp.data_gen.pair_train import get_encode_fn_for_pointwise


JOB_NAME = "mmp_concat_point_train"


def work_fn(job_no):
    n_neg = 5
    st = job_no * 10
    ed = st + 10
    save_dir = path_join("output", "msmarco", "passage", JOB_NAME)
    save_path = os.path.join(save_dir, str(job_no))
    itr = enum_pos_neg_pointwise(range(st, ed), n_neg)
    encode_fn = get_encode_fn_for_pointwise(use_label=True)
    write_records_w_encode_fn(save_path, encode_fn, itr)


def main():
    working_dir = path_join("output", "msmarco", "passage")
    runner = JobRunnerF(working_dir, 13, JOB_NAME, work_fn)
    runner.start()


if __name__ == "__main__":
    main()
