from omegaconf import OmegaConf

from cpath import yconfig_dir_path
from misc_lib import path_join, parallel_run
from trainer_v2.chair_logging import c_log


def get_table_name(method):
    if method.startswith("lkt"):
        return method[4:]
    elif method.startswith("lp_"):
        return method[3:]
    elif method.startswith("lk_"):
        return method[3:]
    else:
        return method


def get_bm25_conf_path(method):
    if method.startswith("lkt"):
        return path_join(yconfig_dir_path, "bm25_resource", "luk_tune.yaml")
    if method.startswith("lp"):
        return path_join(yconfig_dir_path, "bm25_resource", "lucene.yaml")
    else:
        return path_join(yconfig_dir_path, "bm25_resource", "lucene_krovetz.yaml")


def is_bm25t_method(method: str):
    return (method.startswith("mtc")
            or method.startswith("lkt")
            or method.startswith("lk")
            or method.startswith("lp")
            or method == "empty")


def get_bm25t(method):
    from trainer_v2.per_project.transparency.mmp.bm25t_runner.run_rerank4 import get_bm25t_scorer_fn
    table_name = get_table_name(method)

    table_path = path_join("output", "mmp" , "tables", f"{table_name}.tsv")
    conf = OmegaConf.create(
        {
            "bm25conf_path": get_bm25_conf_path(method),
            "table_path": table_path,
            "table_type": "Score",
            "method": method,
            "run_name": method
        }
    )
    score_fn = get_bm25t_scorer_fn(conf)

    def parallel_score_fn(qd_list):
        n_per_job = 1000 * 100
        if len(qd_list) > n_per_job:
            split_n = min(len(qd_list) // n_per_job, 12)
        else:
            split_n = 1

        if split_n > 1:
            c_log.info("Split to %d jobs", split_n)
            ret = parallel_run(qd_list, score_fn, split_n)
        else:
            ret = score_fn(qd_list)
        return ret

    return score_fn
    return parallel_score_fn

