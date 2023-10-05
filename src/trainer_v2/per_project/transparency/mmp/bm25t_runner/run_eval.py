import sys

from omegaconf import OmegaConf

from adhoc.eval_helper.line_format_to_trec_ranked_list import build_ranked_list_from_line_scores_and_eval
from trainer_v2.per_project.transparency.mmp.eval_helper.mmp_eval_line_format import get_line_scores_path


def main():
    run_conf_path = sys.argv[1]

    run_conf = OmegaConf.load(run_conf_path)
    dataset_conf_path = run_conf.dataset_conf_path
    run_name = run_conf.run_name

    dataset_conf = OmegaConf.load(dataset_conf_path)
    dataset_name = dataset_conf.dataset_name
    metric = dataset_conf.metric
    scores_path = get_line_scores_path(run_name, dataset_name)
    judgment_path = dataset_conf.judgment_path
    quad_tsv_path = dataset_conf.rerank_payload_path

    build_ranked_list_from_line_scores_and_eval(
        run_name, dataset_name, judgment_path, quad_tsv_path, scores_path,
        metric)


if __name__ == "__main__":
    main()
