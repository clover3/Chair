from data_generator.data_parser.robust import load_robust_04_query
from data_generator.data_parser.robust2 import load_bm25_best
from data_generator.tokenizer_wo_tf import get_tokenizer
from evals.parse import load_qrels_structured
from tlm.robust.load import load_robust_tokens_for_train, load_robust_tokens_for_predict


def load_tokens():
    tokens_d = load_robust_tokens_for_train()
    tokens_d.update(load_robust_tokens_for_predict(4))
    return tokens_d


class RobustGenCommon:
    def __init__(self, query_type="desc", neg_k=1000):
        qrel_path = "/home/youngwookim/Downloads/rob04-desc/qrels.rob04.txt"
        self.judgement = load_qrels_structured(qrel_path)
        self.queries = load_robust_04_query(query_type)
        self.tokenizer = get_tokenizer()
        self.galago_rank = load_bm25_best()
        self.neg_k = neg_k