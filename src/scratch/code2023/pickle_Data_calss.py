from dataclasses import dataclass

from cache import save_to_pickle, load_from_pickle
from typing import List, Iterable, Callable, Dict, Tuple, Set

from misc_lib import NamedNumber
from trainer_v2.per_project.transparency.mmp.term_effect_measure.core_code import EffectiveRankedListInfo


def main():
    # e_ranked_list_list: List[EffectiveRankedListInfo] = []
    # e = EffectiveRankedListInfo("s", [], [], 1, 1)
    # e_ranked_list_list.append(e)
    nn = NamedNumber(1, "name")
    save_to_pickle(nn, "nn")





if __name__ == "__main__":
    main()