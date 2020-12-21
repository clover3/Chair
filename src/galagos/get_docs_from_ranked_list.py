from typing import List, Dict

from datastore.interface import preload_man, load
from datastore.table_names import TokenizedCluewebDoc
from galagos.parse import load_galago_ranked_list
from galagos.types import SimpleRankedListEntry
from list_lib import lmap, dict_value_map


def get_docs_from_ranked_list(ranked_list: List[SimpleRankedListEntry]) -> List[List[str]]:
    doc_ids = lmap(lambda x: x.doc_id, ranked_list)
    preload_man.preload(TokenizedCluewebDoc, doc_ids)

    def get_tokens(doc_id) -> List[str]:
        return load(TokenizedCluewebDoc, doc_id)

    # tokens_d: Dict[str, List[str]] = load_multiple(TokenizedCluewebDoc, doc_ids, True)

    l : List[List[str]] = []
    cnt_not_found = 0
    for doc_id in doc_ids:
        try:
            r = get_tokens(doc_id)
            l.append(r)
            print(".", end="")
        except KeyError as e:
            cnt_not_found+= 1
            pass
    # l = list(tokens_d.values())
    cnt_not_found = len(doc_ids) - len(l)
    print("done")
    if cnt_not_found:
        print()
        print("not found : ", cnt_not_found)
    return l


def get_docs_from_q_res_path(file_path) -> Dict[str, List[List[str]]]:
    ranked_list_d: Dict[str, List[SimpleRankedListEntry]] = load_galago_ranked_list(file_path)
    return get_docs_from_q_res(ranked_list_d)


def get_docs_from_q_res_path_top_k(file_path, top_k) -> Dict[str, List[List[str]]]:
    ranked_list_d: Dict[str, List[SimpleRankedListEntry]] = load_galago_ranked_list(file_path)
    ranked_list_d = dict_value_map(lambda x:x[:top_k], ranked_list_d)
    return get_docs_from_q_res(ranked_list_d)


def get_docs_from_q_res(ranked_list_d: Dict[str, List[SimpleRankedListEntry]]) -> Dict[str, List[List[str]]]:
    print(len(ranked_list_d))
    return dict_value_map(get_docs_from_ranked_list, ranked_list_d)

