import os

from epath import job_man_dir
from tlm.data_gen.tfrecord_classification import make_classification_from_pair_input_triplet


def main():
    def at_job_man_dir(dir_name):
        return os.path.join(job_man_dir, dir_name)
    make_classification_from_pair_input_triplet(at_job_man_dir('MMD_passage_based_train'),
                                                at_job_man_dir('MMD_dev_sent_split'),
                                                at_job_man_dir('MMD_test_sent_split'),
                                                at_job_man_dir('MMD_passage_based_train_verify'),
                                                )


if __name__ == "__main__":
    main()
