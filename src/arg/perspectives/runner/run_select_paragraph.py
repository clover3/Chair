from arg.perspectives.basic_analysis import load_train_data_point
from arg.perspectives.ranked_list_interface import Q_CONFIG_ID_BM25_10000, StaticRankedListInterface
from arg.perspectives.select_paragraph import select_paragraph_dp_list


def work():
    ci = StaticRankedListInterface(Q_CONFIG_ID_BM25_10000)
    print("load_train_data_point")
    all_data_points = load_train_data_point()
    ##
    print("select paragraph")
    features = select_paragraph_dp_list(ci, all_data_points)


if __name__ == "__main__":
    work()
