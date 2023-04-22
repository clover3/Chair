import time
from collections import Counter, defaultdict
from typing import List, Iterable, Dict, Tuple
from misc_lib import path_join

from adhoc.bm25_class import BM25Bare
from adhoc.kn_tokenizer import KrovetzNLTKTokenizer

#
# f'(x) = (1 + k_1 - k_1 * b * dl / avdl) / (x + k_1 * ((1-b) + b * dl / avdl)).
from dataset_specific.msmarco.passage.passage_resource_loader import enum_grouped, enum_all_when_corpus, FourStr, \
    MMPPosNegSampler
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
        return self.compute_from_tfs(q_terms, t_terms)

    def compute_from_tfs(self, q_terms, t_terms):
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


def tokenize_items(itr, tokenize_fn):
    qid_to_qtf = {}
    pid_to_ptf = {}
    for group in itr:
        for item in group:
            qid, pid, query, text = item
            q_terms = Counter(tokenize_fn(query))
            t_terms = Counter(tokenize_fn(text))
            qid_to_qtf[qid] = q_terms
            pid_to_ptf[pid] = t_terms
    return pid_to_ptf, qid_to_qtf


class GoldPairBasedSampler:
    def __init__(self, mapping):
        # Goal : Enum (q,d) pair where gold label * differ and shallow model has non-zero loss
        #   gold label: score from BERT model
        bm25 = get_mmp_bm25()
        judgment_path = path_join("data", "msmarco", "qrels.train.tsv")
        self.qrels = load_qrels_structured(judgment_path)
        self.bm25t = BM25T(mapping, bm25.core)
        self.grad_computer = BM25T_GradientComputer(bm25.core, mapping)
        c_log.info("init GoldPairBasedSampler done")
        self.time_d = Counter()
        self.pos_neg_sampler = MMPPosNegSampler()

    def compute(self, itr: Iterable[List[FourStr]]) -> dict:
        accum_grad = Counter()
        n_item = 0
        def accumulate(gradient: Counter, direction: int):
            # dL/dw = direction * dy/dw
            for k, v in gradient.items():
                accum_grad[k] += v * direction

            nonlocal n_item
            n_item += 1

        tokenize_fn = self.grad_computer.tokenizer.tokenize_stem
        st = time.time()
        pid_to_ptf, qid_to_qtf = tokenize_items(itr, tokenize_fn)
        elapsed = time.time() - st
        self.time_d['tokenize'] += elapsed

        def sm_scorer(item) -> float:
            qid, pid, query, text = item
            qtf = qid_to_qtf[qid]
            ptf = pid_to_ptf[pid]
            return self.bm25t.score_from_tfs(qtf, ptf)

        def get_gradient(item) -> Counter:
            qid, pid, query, text = item
            qtf = qid_to_qtf[qid]
            ptf = pid_to_ptf[pid]
            gradient = self.grad_computer.compute_from_tfs(qtf, ptf)
            return gradient

        st = time.time()
        for group in itr:
            qid = group[0][0]
            pos, neg = self.pos_neg_sampler.split_pos_neg_entries(group, qid)
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

        elapsed = time.time() - st
        self.time_d['grad'] += elapsed

        return {k: v / n_item for k, v in accum_grad.items()}


QD = Tuple[str, str, Counter, Counter]


class GoldPairBasedSamplerFromTokenized:
    def __init__(self, mapping: Dict[str, Dict[str, float]]):
        # Goal : Enum (q,d) pair where gold label * differ and shallow model has non-zero loss
        #   gold label: score from BERT model
        bm25 = get_mmp_bm25()
        self.bm25t = BM25T(mapping, bm25.core)
        self.grad_computer = BM25T_GradientComputer(bm25.core, mapping)
        self.pos_neg_sampler = MMPPosNegSampler()

    def compute(self, itr: Iterable[List[QD]]) -> Tuple[Dict, Dict]:
        accum_grad = Counter()
        n_item = 0
        def accumulate(gradient: Counter, direction: int):
            # dL/dw = direction * dy/dw
            for k, v in gradient.items():
                accum_grad[k] += v * direction

            nonlocal n_item
            n_item += 1

        def sm_scorer(item) -> float:
            qid, pid, qtf, ptf = item
            return self.bm25t.score_from_tfs(qtf, ptf)

        def get_gradient(item) -> Counter:
            qid, pid, qtf, ptf = item
            gradient = self.grad_computer.compute_from_tfs(qtf, ptf)
            return gradient

        loss_sum = 0
        n_pair = 0
        for group in itr:
            qid = group[0][0]
            pos, neg = self.pos_neg_sampler.split_pos_neg_entries(group, qid)
            for pos_item in pos:
                pos_score = sm_scorer(pos_item)
                pos_grad = get_gradient(pos_item)
                for neg_item in neg:
                    n_pair += 1
                    neg_score = sm_scorer(neg_item)
                    loss = - (pos_score - neg_score)
                    loss_sum += loss
                    # if loss > 0:
                    neg_grad = get_gradient(neg_item)
                    accumulate(pos_grad, -1)
                    accumulate(neg_grad, +1)

                    if 'circa' in pos_grad or 'circa' in neg_grad:
                        print(pos_grad['circa'], neg_grad['circa'], accum_grad['circa'])

        info = {
            'loss_sum': loss_sum,
            'n_pair': n_pair,
            'n_update': n_item / 2,
        }
        out_grad_dict = {k: v / n_item for k, v in accum_grad.items()}
        return out_grad_dict, info

