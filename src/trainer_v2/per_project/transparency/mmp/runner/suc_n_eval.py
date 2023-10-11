import os.path
import re
import sys

from adhoc.eval_helper.pytrec_helper import eval_by_pytrec
from taskman_client.task_proxy import get_task_manager_proxy


def main():
    judgment_path = sys.argv[1]
    ranked_list_path = sys.argv[2]
    try:
        metric = sys.argv[3]
    except IndexError:
        metric = "success_10"
    ret = eval_by_pytrec(
        judgment_path,
        ranked_list_path,
        metric)


    print(f"{metric}:\t{ret}")
    text = os.path.basename(ranked_list_path)
    pattern_list = [
        r'^(.+)_(dev_.+)\.txt$',
        r'^(.+)_(TREC_.+)\.txt$'
    ]

    pattern_found = False
    for pattern in pattern_list:
        match = re.search(pattern, text)
        if match:
            run_name, dataset_name = match.groups()
            print("run_name:", run_name)
            print("dataset_name:", dataset_name)
            proxy = get_task_manager_proxy()
            proxy.report_number(run_name, ret, dataset_name, metric)
            pattern_found = True
            break

    if not pattern_found:
        print("Pattern not found in the text.")




if __name__ == "__main__":
    main()