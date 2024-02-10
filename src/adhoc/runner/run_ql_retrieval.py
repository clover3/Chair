import sys
from dataclasses import dataclass
from typing import Optional

from adhoc.conf_helper import load_omega_config, QLIndexResource
from adhoc.eval_helper.retreival_exp_helper import run_retrieval_eval_report_w_conf
from adhoc.other.ql_retriever_helper import get_ql_retriever_from_conf


@dataclass
class QLRetrievalRunConfigType:
    ql_conf_path: str
    method: str
    run_name: Optional[str]
    dataset_conf_path: str
    outer_batch_size: int = 10000000


def main():
    conf_path = sys.argv[1]
    conf = load_omega_config(conf_path, QLRetrievalRunConfigType)
    ql_conf = load_omega_config(conf.ql_conf_path, QLIndexResource, True)
    retriever = get_ql_retriever_from_conf(ql_conf)
    run_retrieval_eval_report_w_conf(conf, retriever)


if __name__ == "__main__":
    main()
