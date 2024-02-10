import os
import sys
from omegaconf import OmegaConf

from dataset_specific.beir_eval.beir_common import load_beir_dataset
from misc_lib import exist_or_mkdir
from table_lib import tsv_iter
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.misc_common import save_tsv
from trec.trec_parse import load_ranked_list


def build_beir_rerank_conf(dataset_name):
    conf = OmegaConf.create({
        "dataset_name": dataset_name,
        "src_ranked_list": f"output/ranked_list/{dataset_name}_empty.txt",
        "rerank_payload_path": f"output/mmp/rerank/{dataset_name}/corpus.tsv",
    })
    return conf


def main():
    dataset_name = "arguana"
    conf = build_beir_rerank_conf(dataset_name)
    rl = load_ranked_list(conf.src_ranked_list)
    split = "test"
    c_log.info("Loading datasets")
    corpus_d, query_d, _ = load_beir_dataset(dataset_name, split)

    c_log.info("Iterating")
    table = []
    for t in rl:
        query = query_d[t.query_id]
        doc = corpus_d[t.doc_id]
        row = t.query_id, t.doc_id, query, doc
        table.append(row)

    exist_or_mkdir(os.path.dirname(conf.rerank_payload_path))
    save_tsv(table, conf.rerank_payload_path)




if __name__ == "__main__":
    main()