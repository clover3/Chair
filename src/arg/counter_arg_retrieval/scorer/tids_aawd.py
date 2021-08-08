from arg.counter_arg.point_counter.svm_experiment.svm_wrap import SVMWrap
from dataset_specific.aawd.load import get_aawd_binary_train_dev
from misc_lib import tprint


def get_aawd_tids_svm() -> SVMWrap:
    train_x, train_y, dev_x, dev_y = get_aawd_binary_train_dev()
    tprint("training...")
    svm = SVMWrap(train_x, train_y)
    return svm
