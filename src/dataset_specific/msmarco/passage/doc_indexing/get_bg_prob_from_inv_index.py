import pickle
import sys

from cache import load_pickle_from
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator


def main():
    inv_index_path = sys.argv[1]
    bg_prob_save_path = sys.argv[2]

    inv_index: Dict[str, List[Tuple[str, int]]] = load_pickle_from(inv_index_path)
    bg_tf = {}
    for term, postings in inv_index.items():
        n = 0
        for _doc_id, tf in postings:
            n += tf

        bg_tf[term] = n

    ctf = sum(bg_tf.values())
    print("ctf: ", ctf)
    bg_prob: Dict[str, float] = {k: v / ctf for k, v in bg_tf.items()}
    pickle.dump(bg_prob, open(bg_prob_save_path, "wb"))


if __name__ == "__main__":
    main()