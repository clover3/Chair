import os
from typing import Iterable

from arg.counter_arg.header import splits
from cache import load_from_pickle
from cpath import output_path
from evals.trec import TrecRelevanceJudgementEntry, write_trec_relevance_judgement


def get_labels():
    for split in splits:
        job_name = "argu_qck_datagen_{}".format(split)
        candidate_dict, correct_d = load_from_pickle(job_name + "_base_resource")

        for (query_id, candidate_id), correctness in correct_d.items():
            yield query_id, candidate_id, correctness


def get_trec_relevance_judgement() -> Iterable[TrecRelevanceJudgementEntry]:
    for query_id, candidate_id, correctness in get_labels():
        if correctness:
            e = TrecRelevanceJudgementEntry(query_id, candidate_id, int(correctness))
            yield e


def main():
    l = get_trec_relevance_judgement()
    save_path = os.path.join(output_path, "counter_arg", "qrel.txt")
    write_trec_relevance_judgement(l, save_path)


if __name__ == "__main__":
    main()