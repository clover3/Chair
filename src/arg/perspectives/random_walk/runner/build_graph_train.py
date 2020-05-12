from arg.perspectives.random_walk.build_graph_from_tokens import build_co_occur_from_pc_feature
from cache import save_to_pickle, load_from_pickle


def main():
    r = build_co_occur_from_pc_feature(load_from_pickle("pc_train_paas_by_cid"))
    save_to_pickle(r, "pc_train_co_occur")


if __name__ == "__main__":
    main()

