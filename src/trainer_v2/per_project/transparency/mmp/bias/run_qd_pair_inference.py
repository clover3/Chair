import json
import json
import sys
from typing import Iterable, Dict

from omegaconf import OmegaConf

from list_lib import right
from misc_lib import path_join
from table_lib import tsv_iter
from taskman_client.job_group_proxy import SubJobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank import get_scorer_tf_load_model
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def filter_high_score(
        score_fn,
        qid_query_list: list[tuple[str, str]],
        doc_todo: list[tuple[str, str]]) -> Iterable[Dict]:
    query_list = right(qid_query_list)
    for doc_id, doc_text in doc_todo:
        tuple_itr = [(query, doc_text) for query in query_list]
        scores = score_fn(tuple_itr)
        for query_idx, s in enumerate(scores):
            if s > 0:
                qid, query = qid_query_list[query_idx]
                e = {
                    'qid': qid,
                    'query': query,
                    'doc_id': doc_id,
                    'doc_text': doc_text,
                    "score": float(s)
                }
                yield e


def compute_queries_over_docs(config, job_no):
    model_path = config.model_path
    batch_size = 256
    qid_query_list: list[tuple[str, str]] = list(tsv_iter(config.query_path))
    doc_id_passage_list: list[tuple[str, str]] = list(tsv_iter(config.passage_path))
    save_path = path_join(config.save_dir, f"{job_no}.jsonl")
    f = open(save_path, "w")
    n_per_job = config.n_per_job
    st = job_no * n_per_job
    ed = st + n_per_job

    strategy = get_strategy()
    with strategy.scope():
        c_log.info("Building scorer")
        score_fn = get_scorer_tf_load_model(model_path, batch_size)

    doc_todo: list[tuple[str, str]] = [doc_id_passage_list[doc_idx] for doc_idx in range(st, ed)]
    matching_pairs: Iterable[Dict] = filter_high_score(score_fn, qid_query_list, doc_todo)

    for d in matching_pairs:
        j = json.dumps(d)
        f.write(j + "\n")
        f.flush()


def main():
    conf_path = sys.argv[1]
    job_no = int(sys.argv[2])
    config = OmegaConf.load(conf_path)
    c_log.info("Job %d", job_no)
    job_name = config.job_name
    with SubJobContext(job_name, job_no, config.max_job):
        compute_queries_over_docs(config, job_no)


if __name__ == "__main__":
    main()
