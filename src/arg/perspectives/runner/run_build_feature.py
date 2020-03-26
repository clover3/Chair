from functools import partial

from arg.perspectives.basic_analysis import load_train_data_point
from arg.perspectives.build_feature import build_binary_feature, build_weighted_feature
from arg.perspectives.clueweb_galago_db import DocGetter
from arg.perspectives.ranked_list_interface import Q_CONFIG_ID_BM25_10000, DynRankedListInterface, make_doc_query
from cache import save_to_pickle
from misc_lib import parallel_run


def work():
    opt = "binary"
    ci = DynRankedListInterface(make_doc_query, Q_CONFIG_ID_BM25_10000)
    doc_getter = DocGetter()
    print("load_train_data_point")
    all_data_points = load_train_data_point()
    ##
    print("")
    if opt == "weighted":
        features = parallel_run(all_data_points, build_weighted_feature, 1000)
        save_to_pickle(features, "pc_train_features")
    elif opt == "binary":
        build_binary_feature_fn = partial(build_binary_feature, ci)
        features = build_binary_feature_fn(all_data_points)
        #features = parallel_run(all_data_points, build_binary_feature_fn, 1000)

        save_to_pickle(features, "pc_train_features_binary")
    else:
        assert False
        ###
    print("{} build from {}".format(len(features), len(all_data_points)))


if __name__ == "__main__":
    work()
