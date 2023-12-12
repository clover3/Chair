import sys
from typing import List, Iterable, Callable, Dict, Tuple, Set

from misc_lib import get_second
from trainer_v2.per_project.transparency.mmp.bm25t_helper import load_align_scores


def main():
    table_path = sys.argv[1]
    table: Dict[str, Dict[str, float]] = load_align_scores(table_path)

    q_term_cnt = []
    for q_term, entries in table.items():
        q_term_cnt.append((q_term, len(entries)))

    q_term_cnt.sort(key=get_second, reverse=True)
    for q_term, cnt in q_term_cnt[:100]:
        print(q_term, cnt)


if __name__ == "__main__":
    main()
