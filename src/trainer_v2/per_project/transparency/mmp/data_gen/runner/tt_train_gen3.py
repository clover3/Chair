import os

from job_manager.job_runner_with_server import JobRunnerF
from misc_lib import path_join
from tf_util.record_writer_wrap import write_records_w_encode_fn
from trainer_v2.per_project.transparency.mmp.data_enum import enum_pos_neg_sample
from trainer_v2.per_project.transparency.mmp.data_gen.tt_train_gen import get_encode_fn_for_word_encoder, \
    get_encode_fn_for_word_encoder_qtw


def work_fn(job_no):
    st = job_no * 10
    ed = st + 10
    save_dir = path_join("output", "msmarco", "passage", "train_tt3")
    save_path = os.path.join(save_dir, str(job_no))
    itr = enum_pos_neg_sample(range(st, ed))
    encode_fn = get_encode_fn_for_word_encoder_qtw()
    write_records_w_encode_fn(save_path, encode_fn, itr)


def main():
    working_dir = path_join("output", "msmarco", "passage")
    runner = JobRunnerF(working_dir, 11, "train_tt3", work_fn)
    runner.start()



if __name__ == "__main__":
    main()