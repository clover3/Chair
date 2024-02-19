from typing import Any

from dataset_specific.beir_eval.beir_common import beir_mb_dataset_list, beir_dataset_list_A, beir_dataset_name_map
from tab_print import print_table, transpose
from taskman_client.named_number_proxy import NamedNumberProxy
from trainer_v2.chair_logging import c_log
from table_lib import TablePrintHelper, DictCache


def index_by_name_condition(ret, target_field):
    d = {}
    for e in ret:
        if e['field'] == target_field:
            key = e['name'], e['condition']
            if key in d:
                c_log.warning("key=%s is duplicated originally %s, replace with %s",
                           str(key), str(d[key]), str(e['number']))
            d[key] = e['number']
    return d


def print_scores_by_asking_server(condition_list, method_list, method_name_map):
    search = NamedNumberProxy()
    target_field = "ndcg_cut_10"

    def load_method_relate_scores(method):
        ret = search.search_numbers(method)
        print(ret)
        d: dict[str, Any] = {}
        for e in ret:
            if e['field'] == target_field and e['name'] == method:
                key = e['condition']
                if key in d:
                    c_log.warning("key=%s is duplicated originally %s, replace with %s",
                                  str(key), str(d[key]), str(e['number']))
                d[e['condition']] = e['number']
        return d

    col_cache = DictCache(load_method_relate_scores)

    def get_score(row_key, col_key):
        per_col_d = col_cache.get_val(col_key)
        return per_col_d[row_key]

    printer = TablePrintHelper(
        method_list,
        condition_list,
        method_name_map,
        beir_dataset_name_map,
        get_score,
        "Method",
    )
    print_table(printer.get_table())


def main():
    method_list = [
        "empty",
        "rr_mtc6_pep_tt17_10000",
        "rr_ce_msmarco_mini_lm"
    ]
    method_name_map = {
        "empty": "BM25",
        "rr_mtc6_pep_tt17_10000": "BM25T",
        "rr_ce_msmarco_mini_lm": "CrossEncoder"
    }
    condition_list = beir_dataset_list_A

    print_scores_by_asking_server(condition_list, method_list, method_name_map)


if __name__ == "__main__":
    main()
