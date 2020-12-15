import os
from typing import Iterable, Tuple

from arg.counter_arg.header import splits
from arg.qck.qrel_helper import get_trec_relevance_judgement
from cache import load_from_pickle
from cpath import output_path
from trec.trec_parse import write_trec_relevance_judgement


def get_labels() -> Iterable[Tuple[str, str, int]]:
    for split in splits:
        job_name = "argu_qck_datagen_{}".format(split)
        candidate_dict, correct_d = load_from_pickle(job_name + "_base_resource")

        for (query_id, candidate_id), correctness in correct_d.items():
            yield query_id, candidate_id, correctness


def main():
    label_itr = get_labels()
    l = get_trec_relevance_judgement(label_itr)
    save_path = os.path.join(output_path, "counter_arg", "qrel.txt")
    write_trec_relevance_judgement(l, save_path)


if __name__ == "__main__":
    main()