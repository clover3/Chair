from typing import List, Dict

from datastore.interface import load_multiple
from datastore.table_names import TokenizedCluewebDoc
from list_lib import lmap
from misc_lib import get_duplicate_list


def remove_duplicate(doc_id_list: List[str]) -> List[str]:
    docs_d: Dict[str, List[str]] = load_multiple(TokenizedCluewebDoc, doc_id_list, True)
    hashes = lmap(doc_hash, [docs_d[doc_id] if doc_id in docs_d else None for doc_id in doc_id_list])
    duplicate_indice = get_duplicate_list(hashes)
    non_duplicate = list([doc_id_list[i] for i in range(len(doc_id_list)) if i not in duplicate_indice])
    return non_duplicate


def doc_hash(doc: List[str]):
    if doc is None:
        return " "
    else:
        return " ".join(doc)