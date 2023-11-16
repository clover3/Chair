import sys
from multiprocessing import Pool

from omegaconf import OmegaConf

from adhoc.bm25_class import BM25
from dataset_specific.msmarco.passage.load_term_stats import load_msmarco_passage_term_stat
from trainer_v2.per_project.transparency.mmp.bm25t import BM25T
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank_w_conf import run_rerank_with_conf_common
from trainer_v2.per_project.transparency.mmp.retrieval_run.run_bm25t import load_table_from_conf, to_value_dict


def parallel_run(input_list, list_fn, split_n):
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            print(i, i + n)
            yield l[i:i + n]

    p = Pool(split_n)

    item_per_job = (len(input_list) + split_n - 1) // split_n
    print("item_per_job", item_per_job)
    l_args = chunks(input_list, item_per_job)

    result_list_list = p.map(list_fn, l_args)

    result = []
    for result_list in result_list_list:
        result.extend(result_list)
    return result


def get_bm25t_scorer_fn(conf):
    mapping = load_table_from_conf(conf)
    value_mapping = to_value_dict(mapping, conf.mapping_val)

    cdf, df = load_msmarco_passage_term_stat()
    bm25 = BM25(df, cdf, 25, 0.1, 100, 1.4)
    bm25t = BM25T(value_mapping, bm25.core)

    def parallel_score_fn(qd_list):
        ret = parallel_run(qd_list, bm25t.score_batch, 20)
        return ret

    return parallel_score_fn


def main():
    get_scorer_fn = get_bm25t_scorer_fn
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    assert int(conf.outer_batch_size) > 1
    run_rerank_with_conf_common(conf, get_scorer_fn)


if __name__ == "__main__":
    main()
