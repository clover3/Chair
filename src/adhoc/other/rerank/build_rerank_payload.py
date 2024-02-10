import os.path
import sys
from omegaconf import OmegaConf

from table_lib import tsv_iter
from trainer_v2.per_project.transparency.misc_common import save_tsv
from trec.trec_parse import load_ranked_list
from cpath import output_path
from misc_lib import path_join, exist_or_mkdir


def main():
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    rl = load_ranked_list(conf.src_ranked_list)
    query_d = {k: v for k, v in tsv_iter(conf.queries_path)}
    corpus_d = {k: v for k, v in tsv_iter(conf.corpus_path)}

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