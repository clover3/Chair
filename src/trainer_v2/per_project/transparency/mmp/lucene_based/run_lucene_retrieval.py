import sys
from omegaconf import OmegaConf
from adhoc.eval_helper.retreival_exp_helper import run_retrieval_eval_report_w_conf
from adhoc.other.pylucene_dev.lucene_searcher import get_lucene_searcher


def main():
    dataset_conf_path = sys.argv[1]
    d = {"dataset_conf_path": dataset_conf_path}
    conf = OmegaConf.create(d)
    # conf.run_name = "pylucene_0904"
    conf.method = "pylucene_bm25"
    retriever = get_lucene_searcher()
    run_retrieval_eval_report_w_conf(conf, retriever)


if __name__ == "__main__":
    main()
