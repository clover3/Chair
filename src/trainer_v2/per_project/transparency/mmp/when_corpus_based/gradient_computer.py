from collections import Counter, defaultdict
from typing import List, Iterable, Dict
from misc_lib import path_join

from adhoc.bm25_class import BM25Bare
from adhoc.kn_tokenizer import KrovetzNLTKTokenizer

#
# f'(x) = (1 + k_1 - k_1 * b * dl / avdl) / (x + k_1 * ((1-b) + b * dl / avdl)).
from dataset_specific.msmarco.passage.passage_resource_loader import enum_grouped, enum_all_when_corpus, FourStr
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.bm25t import BM25T
from trainer_v2.per_project.transparency.mmp.when_corpus_based.when_bm25t import build_table, get_mmp_bm25
from trec.qrel_parse import load_qrels_structured


def bm25_term_derivation(cur_tf, k1, b, dl, avdl, ):
    denom = (1 + k1 - k1 * b * dl / avdl)
    nom = (cur_tf + k1 * ((1 - b) + b * dl / avdl))
    return denom / nom


class BM25T_GradientComputer:
    def __init__(self, bm25_bare: BM25Bare, mapping):
        self.bm25_bare = bm25_bare
        self.mapping = mapping
        self.tokenizer = KrovetzNLTKTokenizer()

    def compute_from_two_text(self, query, text) -> Counter:
        q_terms = self.tokenizer.tokenize_stem(query)
        t_terms = self.tokenizer.tokenize_stem(text)
        q_tf = Counter(q_terms)
        t_tf = Counter(t_terms)
        dl = sum(t_tf.values())

        gradient = Counter()
        for q_term, q_cnt in q_tf.items():
            if q_term not in self.mapping:
                continue
            translation_term_set: Dict[str, float] = self.mapping[q_term]

            overlap_terms = []
            for k in t_tf:
                if k in translation_term_set:
                    overlap_terms.append(k)
            if not overlap_terms:
                continue
            raw_cnt = t_tf[q_term]
            dy_dx_part1 = self.bm25_bare.term_idf_factor(q_term)
            k1 = self.bm25_bare.k1
            b = self.bm25_bare.b
            avdl = self.bm25_bare.avdl
            dy_dx_part2 = bm25_term_derivation(raw_cnt, k1, b, dl, avdl)
            dy_dx_factor = dy_dx_part1 * dy_dx_part2
            for t in overlap_terms:
                gradient[t] += dy_dx_factor * t_tf[t]
        return gradient


class GoldPairBasedSampler:
    def __init__(self, mapping):
        # Goal : Enum (q,d) pair where gold label* differ and shallow model has non-zero loss
        #   gold lable: score from BERT model
        bm25 = get_mmp_bm25()
        judgment_path = path_join("data", "msmarco", "qrels.train.tsv")
        self.qrels = load_qrels_structured(judgment_path)
        self.bm25t = BM25T(mapping, bm25.core)
        self.grad_computer = BM25T_GradientComputer(bm25.core, mapping)
        c_log.info("init GoldPairBasedSampler done")

    def compute(self, itr: Iterable[List[FourStr]]) -> dict:
        def split_pos_neg_entries(qid, entries):
            pos_doc_ids = []
            for doc_id, score in self.qrels[qid].items():
                if score > 0:
                    pos_doc_ids.append(doc_id)

            pos_doc = []
            neg_doc = []
            for e in entries:
                qid, pid, query, text = e
                if pid in pos_doc_ids:
                    pos_doc.append(e)
                else:
                    neg_doc.append(e)
            return pos_doc, neg_doc


        def get_gradient(item) -> Counter:
            qid, pid, query, text = item
            gradient = self.grad_computer.compute_from_two_text(query, text)
            return gradient

        accum_grad = Counter()
        n_item = 0
        def accumulate(gradient: Counter, direction: int):
            # dL/dw = direction * dy/dw
            for k, v in gradient.items():
                accum_grad[k] += v * direction

            nonlocal n_item
            n_item += 1

        def sm_scorer(item) -> float:
            qid, pid, query, text = item
            return self.bm25t.score(query, text)

        for group in itr:
            qid = group[0][0]
            pos, neg = split_pos_neg_entries(qid, group)
            for pos_item in pos:
                pos_score = sm_scorer(pos_item)
                pos_grad = get_gradient(pos_item)
                for neg_item in neg:
                    neg_score = sm_scorer(neg_item)
                    loss = max(1 - (pos_score - neg_score), 0)
                    if loss > 0:
                        neg_grad = get_gradient(neg_item)
                        accumulate(pos_grad, -1)
                        accumulate(neg_grad, +1)

        return {k: v / n_item for k, v in accum_grad.items()}


def sample_pairs_with_gold_label():
    mapping = defaultdict(dict)
    mapping['when'] = build_table()

    itr = enum_all_when_corpus()
    itr = enum_grouped(itr)
