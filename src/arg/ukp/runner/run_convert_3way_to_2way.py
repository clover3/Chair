import os

from base_type import FileName
from cpath import pjoin
from data_generator.job_runner import sydney_working_dir
from misc_lib import exist_or_mkdir, get_dir_files
from tlm.data_gen.convert_3way_to_2way import convert


def run():
    in_dir_name = FileName("ukp_paragraph_tfrecord_dev_abortion")
    out_dir_name = FileName("ukp_paragraph_tfrecord_dev_abortion_2way")
    run_dir(in_dir_name, out_dir_name)

    in_dir_name = FileName("ukp_paragraph_tfrecord_train_abortion")
    out_dir_name = FileName("ukp_paragraph_tfrecord_train_abortion_2way")
    run_dir(in_dir_name, out_dir_name)


def run_dir(in_dir_name: FileName, out_dir_name: FileName):
    in_dir = pjoin(sydney_working_dir, in_dir_name)
    out_dir = pjoin(sydney_working_dir, out_dir_name)
    exist_or_mkdir(out_dir)

    for file_path in get_dir_files(in_dir):
        name = FileName(os.path.basename(file_path))
        out_path = pjoin(out_dir, name)
        convert(file_path, out_path)


if __name__ == '__main__':
    run()

