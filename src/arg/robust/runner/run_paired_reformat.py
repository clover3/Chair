import os

from epath import job_man_dir
from misc_lib import exist_or_mkdir
from tlm.data_gen.reformat_paired_data import convert_to_unpaired


def main():
    input_dir = os.path.join(job_man_dir, "robust_all_passage")
    ouput_dir = os.path.join(job_man_dir, "robust_all_passage_unpaired_2")
    exist_or_mkdir(ouput_dir)

    file_names = ["301", "351", "401", "601", "651"]

    from pathos.multiprocessing import ProcessingPool as Pool

    def fn(name):
        convert_to_unpaired(os.path.join(input_dir, name), os.path.join(ouput_dir, name))

    p = Pool(5, daemon=True)
    args = file_names
    result_list_list = p.map(fn, args)


if __name__ == "__main__":
    main()