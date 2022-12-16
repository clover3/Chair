from typing import Dict
from contradiction.medical_claims.token_tagging.path_helper import get_sbl_vak_qrel_path, get_save_path2
from taskman_client.task_proxy import get_task_manager_proxy
from trec.trec_eval_wrap_fn import run_trec_eval_parse


def do_ecc_eval_w_trec_eval(run_name, label_type, do_report=False) -> Dict:
    prediction_path = get_save_path2(run_name, label_type)
    qrel_path = get_sbl_vak_qrel_path()
    target_metric = "map"
    ret = run_trec_eval_parse(prediction_path, qrel_path, target_metric)

    if do_report:
        proxy = get_task_manager_proxy()
        for k, v in ret.items():
            proxy.report_number(run_name, v, "", "ECC_MAP")
    return ret


def main():
    ret = do_ecc_eval_w_trec_eval("exact_match", "mismatch", True)
    print(ret)


if __name__ == "__main__":
    main()
