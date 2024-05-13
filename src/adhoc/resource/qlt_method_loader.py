from omegaconf import OmegaConf
from cpath import yconfig_dir_path
from misc_lib import path_join, parallel_run


def get_table_name(method):
    if method.startswith("qlt_"):
        return method[4:]
    else:
        return method


def get_ql_conf_path():
    return path_join(yconfig_dir_path, "bm25_resource", "lucene_krovetz_ql.yaml")


def is_qlt_method(method: str):
    return method.startswith("qlt_")



def get_ql(method):
    from trainer_v2.per_project.transparency.mmp.bm25t.ql_rerank import get_ql_scorer_fn
    conf = OmegaConf.create(
        {
            "ql_conf_path": get_ql_conf_path(),
            "table_type": "Score",
            "method": method,
            "run_name": method
        }
    )
    score_fn = get_ql_scorer_fn(conf)
    return score_fn


def get_qlt(method):
    table_name = get_table_name(method)
    table_path = path_join("output", "mmp", "tables", f"{table_name}.tsv")
    from trainer_v2.per_project.transparency.mmp.bm25t.ql_rerank import get_ql_t_scorer_fn

    if table_name == "empty":
        table_path = "none"

    conf = OmegaConf.create(
        {
            "ql_conf_path": get_ql_conf_path(),
            "table_path": table_path,
            "table_type": "Score",
            "method": method,
            "run_name": method
        }
    )
    score_fn = get_ql_t_scorer_fn(conf)
    return score_fn