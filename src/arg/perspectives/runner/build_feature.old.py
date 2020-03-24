from functools import partial

from arg.perspectives.basic_analysis import load_train_data_point
from arg.perspectives.build_feature import build_binary_feature, build_weighted_feature
from arg.perspectives.collection_interface import CollectionInterface
from cache import save_to_pickle
from misc_lib import parallel_run


def work():
    opt = "binary"
    ci = CollectionInterface()

    all_data_points = load_train_data_point()

    if opt == "weighted":
        features = parallel_run(all_data_points, build_weighted_feature, 1000)
        save_to_pickle(features, "pc_train_features")
    elif opt == "binary":
        build_binary_feature_fn = partial(build_binary_feature, ci)
        features = parallel_run(all_data_points, build_binary_feature_fn, 1000)
        save_to_pickle(features, "pc_train_features_binary")
    else:
        assert False

    print("{} build from {}".format(len(features), len(all_data_points)))


if __name__ == "__main__":
    work()
