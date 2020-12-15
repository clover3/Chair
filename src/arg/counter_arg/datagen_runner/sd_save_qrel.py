import os

from arg.counter_arg.header import splits
from arg.counter_arg.same_debate import load_base_resource
from arg.qck.qrel_helper import get_trec_relevance_judgement
from cpath import output_path
from trec.trec_parse import write_trec_relevance_judgement


def get_labels():
    for split in splits:
        candidate_dict, correct_d = load_base_resource(split)
        for (query_id, candidate_id), correctness in correct_d.items():
            if correctness:
                yield query_id, candidate_id, correctness


def main():
    label_itr = get_labels()
    l = get_trec_relevance_judgement(label_itr)
    save_path = os.path.join(output_path, "counter_arg", "sd_qrel_small.txt")
    write_trec_relevance_judgement(l, save_path)


if __name__ == "__main__":
    main()