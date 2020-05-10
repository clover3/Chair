import os
from typing import List, Tuple

from arg.perspectives.load import load_train_claim_ids, load_dev_claim_ids
from cache import save_to_pickle
from sydney_clueweb.clue_path import index_name_list


def collect_from(claim_ids, dir_path):
    disk_name = index_name_list[0]

    def read_file(file_path) -> List[Tuple[str, str]]:
        r = []
        for line in open(file_path):
            if line.strip():
                try:
                    term, score = line.split("\t")
                    r.append((term, score))
                except ValueError:
                    print(line)
                    raise
        return r

    d = {}
    for claim_id in claim_ids:
        file_path = os.path.join(dir_path, "{}_{}.txt".format(disk_name, claim_id))
        try:
            d[claim_id] = read_file(file_path)
        except FileNotFoundError:
            pass

    return d


def main():
    claim_ids = load_train_claim_ids()
    dir_path = "/mnt/nfs/work3/youngwookim/data/perspective/train_claim_rm3"
    term_d = collect_from(claim_ids, dir_path)
    save_to_pickle(term_d, "perspective_train_claim_rm")


    claim_ids = load_dev_claim_ids()
    dir_path = "/mnt/nfs/work3/youngwookim/data/perspective/dev_claim_rm3"
    term_d = collect_from(claim_ids, dir_path)
    save_to_pickle(term_d, "perspective_dev_claim_rm")


if __name__ == "__main__":
    main()