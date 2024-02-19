import csv
import os
import sys
from omegaconf import OmegaConf

from adhoc.conf_helper import create_omega_config
from dataset_specific.beir_eval.beir_common import load_beir_dataset, beir_dataset_list_A
from misc_lib import exist_or_mkdir
from table_lib import tsv_iter
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.misc_common import save_tsv
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank_w_conf import RerankDatasetConf
from trec.trec_parse import load_ranked_list, load_ranked_list_grouped


def build_beir_rerank_conf(dataset_name) -> RerankDatasetConf:
    conf = create_omega_config({
        "dataset_name": dataset_name,
        "src_ranked_list": f"output/ranked_list/{dataset_name}_empty.txt",
        "rerank_payload_path": f"output/mmp/rerank100/{dataset_name}/corpus.csv",
        "metric": "ndcg_cut_10",

    }, RerankDatasetConf)
    return conf


def check_tabs():
    n_per_query = 100
    for dataset_name in beir_dataset_list_A:
        c_log.info("Working for %s", dataset_name)
        conf = build_beir_rerank_conf(dataset_name)
        rlg = load_ranked_list_grouped(conf.src_ranked_list)
        split = "test"
        c_log.info("Loading datasets")
        corpus_d, query_d, _ = load_beir_dataset(dataset_name, split)

        c_log.info("Iterating")
        has_tab = False

        for qid, entries in rlg.items():
            entries.sort(key=lambda x: x.score, reverse=True)
            entries = entries[:n_per_query]
            for t in entries:
                query = query_d[t.query_id]
                doc = corpus_d[t.doc_id]
                doc_text = doc['title'] + " " + doc['text']
                if "\t" in query or "\t" in doc_text:
                    has_tab = True
                    break
            if has_tab:
                break
        print("has_tab", has_tab)



def main():
    n_per_query = 100
    for dataset_name in beir_dataset_list_A:
        c_log.info("Working for %s", dataset_name)
        conf = build_beir_rerank_conf(dataset_name)

        if os.path.exists(conf.rerank_payload_path):
            c_log.info("Skip %s", dataset_name)
            continue
        rlg = load_ranked_list_grouped(conf.src_ranked_list)
        split = "test"
        c_log.info("Loading datasets")
        corpus_d, query_d, _ = load_beir_dataset(dataset_name, split)

        c_log.info("Iterating")
        table = []
        for qid, entries in rlg.items():
            entries.sort(key=lambda x: x.score, reverse=True)
            entries = entries[:n_per_query]
            for t in entries:
                query = query_d[t.query_id]
                doc = corpus_d[t.doc_id]
                doc_text = doc['title'] + " " + doc['text']
                assert isinstance(query, str)
                assert isinstance(doc_text, str)
                row = t.query_id, t.doc_id, query, doc_text
                table.append(row)

        exist_or_mkdir(os.path.dirname(conf.rerank_payload_path))
        csv_writer = csv.writer(open(conf.rerank_payload_path, "w", newline='', encoding="utf-8"))
        csv_writer.writerows(table)


if __name__ == "__main__":
    main()