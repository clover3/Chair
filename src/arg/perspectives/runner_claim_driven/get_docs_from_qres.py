from base_type import FilePath
from cache import save_to_pickle
from galagos.get_docs_from_ranked_list import get_docs_from_q_res_path


def main():
    ranked_list_dev_path = FilePath("/mnt/nfs/work3/youngwookim/data/perspective/dev_claim/q_res_100")
    r = get_docs_from_q_res_path(ranked_list_dev_path)
    save_to_pickle(r, "dev_claim_docs")

    ranked_list_train_path = FilePath("/mnt/nfs/work3/youngwookim/data/perspective/train_claim/q_res_100")
    r = get_docs_from_q_res_path(ranked_list_train_path )
    save_to_pickle(r, "train_claim_docs")


if __name__ == "__main__":
    main()


