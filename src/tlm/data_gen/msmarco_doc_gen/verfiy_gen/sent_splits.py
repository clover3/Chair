import os

from epath import job_man_dir
from tlm.data_gen.tfrecord_classification import make_classification_from_single_input_triplet


def main():
    def at_job_man_dir(dir_name):
        return os.path.join(job_man_dir, dir_name)
    make_classification_from_single_input_triplet(at_job_man_dir('MMD_train_A'),
                                                at_job_man_dir('MMD_dev_sent_split'),
                                                at_job_man_dir('MMD_test_sent_split'),
                                                at_job_man_dir('MMD_train_A_sent_verfiy'),
                                                )


if __name__ == "__main__":
    main()
