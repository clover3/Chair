from dataset_specific.beir_eval.beir_common import beir_mb_dataset_list
from tab_print import print_table
from taskman_client.named_number_proxy import NamedNumberProxy
from trainer_v2.chair_logging import c_log


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


def main():
    search = NamedNumberProxy()
    method1 = "empty"
    method2 = "mtc6_pep_tt17_30000"
    target_field = "NDCG@10"
    condition_list = beir_mb_dataset_list

    head = ["method"] + condition_list
    table = [head]
    for method in [method1, method2]:
        ret = search.search_numbers(method)
        print(ret)
        d = index_by_name_condition(ret, target_field)
        row = [method]
        for c in condition_list:
            key = method, c
            try:
                val = d[key]
                row.append(str(val))
            except KeyError:
                row.append("-")
        table.append(row)
    print_table(table)


if __name__ == "__main__":
    main()