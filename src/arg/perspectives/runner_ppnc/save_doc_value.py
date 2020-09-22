from typing import Dict, Tuple

from arg.perspectives.ppnc.get_doc_value import load_passage_score_d
from cache import save_to_pickle, load_from_pickle


def main():
    score_d = load_passage_score_d("qcknc_val", "train_baseline")
    save_to_pickle(score_d, "cppnc_doc_value_val")


def load_cppnc_doc_value_val() -> Dict[Tuple[str, str, int], float]:
    return load_from_pickle("cppnc_doc_value_val")


if __name__ == "__main__":
    main()
