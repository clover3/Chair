from arg.perspectives.basic_analysis import load_train_data_point
from arg.perspectives.collection_based_classifier import build_weighted_feature, build_binary_feature
from cache import save_to_pickle
from misc_lib import parallel_run


def work():
    opt = "binary"
    all_data_points = load_train_data_point()
    if opt == "weighted":
        features = parallel_run(all_data_points, build_weighted_feature, 1000)
        save_to_pickle(features, "pc_train_features")
    elif opt == "binary":
        features = parallel_run(all_data_points, build_binary_feature, 1000)
        save_to_pickle(features, "pc_train_features_binary")
    else:
        assert False

    print("{} build from {}".format(len(features), len(all_data_points)))


if __name__ == "__main__":
    work()
