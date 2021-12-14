import os
from typing import List, Dict

from cpath import output_path
from trec.trec_parse import load_ranked_list_grouped, write_trec_ranked_list_entry
from trec.types import TrecRankedListEntry


def slice_for_run(run_name, k):
    ranked_list_path = os.path.join(output_path, "ca_building",
                                    "passage_ranked_list", '{}.txt'.format(run_name))
    rl: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(ranked_list_path)

    new_ranked_list = []
    for key, value in rl.items():
        new_ranked_list.extend(value[:k])

    save_path = os.path.join(output_path, "ca_building",
                                    "passage_ranked_list_sliced", '{}.txt'.format(run_name))

    write_trec_ranked_list_entry(new_ranked_list, save_path)


def main():
    k = 20
    for j in range(6, 10):
        slice_for_run("PQ_{}".format(j), k)


if __name__ == "__main__":
    main()