import os

from job_manager.job_runner_with_server import JobRunnerF
from misc_lib import path_join
from tf_util.record_writer_wrap import write_records_w_encode_fn
from trainer_v2.per_project.transparency.mmp.data_enum import enum_pos_neg_multi_sample
from trainer_v2.per_project.transparency.mmp.data_gen.tt_train_gen import get_encode_fn_for_word_encoder, \
    get_encode_fn_for_word_encoder_qtw


data_name = "train_tt4"
def work_fn(job_no):
    st = job_no * 10
    ed = st + 10
    save_dir = path_join("output", "msmarco", "passage", data_name)
    save_path = os.path.join(save_dir, str(job_no))
    itr = enum_pos_neg_multi_sample(range(st, ed), 10)
    encode_fn = get_encode_fn_for_word_encoder_qtw()
    write_records_w_encode_fn(save_path, encode_fn, itr)


def main():
    working_dir = path_join("output", "msmarco", "passage")
    runner = JobRunnerF(working_dir, 11, data_name, work_fn)
    runner.start()



if __name__ == "__main__":
    main()