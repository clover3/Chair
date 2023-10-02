import logging
from collections import Counter, defaultdict
from typing import Dict

from adhoc.bm25 import BM25_verbose
from adhoc.kn_tokenizer import KrovetzNLTKTokenizer
from dataset_specific.msmarco.passage.load_term_stats import load_msmarco_passage_term_stat
from table_lib import tsv_iter
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.when_corpus_based.when_bm25t import build_table_when_avg
from cpath import output_path
from misc_lib import path_join
from trec.qrel_parse import load_qrels_structured


def main():
    c_log.setLevel(logging.DEBUG)
    mapping = defaultdict(dict)
    mapping['when'] = build_table_when_avg()
    cdf, df = load_msmarco_passage_term_stat()
    avdl = 25
    N = cdf
    k1 = 0.1
    k2 = 0
    b = 0.1
    judgment_path = path_join("data", "msmarco", "qrels.train.tsv")
    qrels = load_qrels_structured(judgment_path)

    tokenizer = KrovetzNLTKTokenizer()
    quad_tsv_path = path_join(output_path, "msmarco", "passage", "when_full", "0")
    itr = tsv_iter(quad_tsv_path)
    for qid, doc_id, query, text in itr:
        qrel_d = qrels[qid]
        is_rel = doc_id in qrel_d and qrel_d[doc_id]
        s_is_rel = "Rel" if is_rel else "NonRel"

        q_terms = tokenizer.tokenize_stem(query)
        t_terms = tokenizer.tokenize_stem(text)
        q_tf = Counter(q_terms)
        t_tf = Counter(t_terms)
        dl = sum(t_tf.values())
        score_sum = 0
        for q_term, q_cnt in q_tf.items():
            translation_term_set: Dict[str, float] = mapping[q_term]
            expansion_tf = 0
            for t, cnt in t_tf.items():
                if t in translation_term_set:
                    expansion_tf += cnt * translation_term_set[t]
                    c_log.debug(f"{s_is_rel} matched {t} has {translation_term_set[t]}")

            raw_cnt = t_tf[q_term]
            tf_sum = expansion_tf + raw_cnt

            t = BM25_verbose(f=tf_sum,
                         qf=q_cnt,
                         df=df[q_term],
                         N=N,
                         dl=dl,
                         avdl=avdl,
                         b=b,
                         my_k1=k1,
                         my_k2=k2
                         )
            if expansion_tf:
                c_log.debug(f"tf_sum={expansion_tf}+{raw_cnt}, adding {t} to total")

            score_sum += t




if __name__ == "__main__":
    main()

