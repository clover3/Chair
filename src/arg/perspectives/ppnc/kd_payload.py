from typing import List, Dict, Tuple

from arg.perspectives.ppnc.decl import ClaimPassages
from cache import load_from_pickle
from list_lib import lmap


def load_from_old_to_new(save_name) -> List[ClaimPassages]:
    data = load_from_pickle(save_name)
    entries: List[Tuple[Dict, List[Tuple[List[str], float]]]] = data[0]
    return lmap(lambda x: ClaimPassages(x), entries)


def load_dev_payload() -> List[ClaimPassages]:
    save_name = "pc_dev_a_passages"
    return load_from_old_to_new(save_name)

