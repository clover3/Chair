import sys

from omegaconf import OmegaConf

from adhoc.eval_helper.line_format_to_trec_ranked_list import build_ranked_list_from_qid_pid_scores
from adhoc.eval_helper.pytrec_helper import eval_by_pytrec_json_qrel


def main():
    run_conf_path = sys.argv[1]

    run_conf = OmegaConf.load(run_conf_path)
    dataset_conf_path = run_conf.dataset_conf_path
    dataset_conf = OmegaConf.load(dataset_conf_path)

    quad_tsv_path = dataset_conf.rerank_payload_path
    build_ranked_list_from_qid_pid_scores(
        quad_tsv_path,
        run_conf.run_name,
        run_conf.ranked_list_path,
        run_conf.scores_path)
    ret = eval_by_pytrec_json_qrel(
        dataset_conf.judgment_path,
        run_conf.ranked_list_path,
        dataset_conf.metric)
    print(ret)

if __name__ == "__main__":
    main()
