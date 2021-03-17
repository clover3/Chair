import collections
from typing import Iterator, Iterable

from arg.qck.decl import QCKQuery, QCKCandidate, get_light_qckquery, get_light_qckcandidate
from data_generator.data_parser.robust import load_robust_04_query
from data_generator.data_parser.robust2 import load_bm25_best
from data_generator.tokenizer_wo_tf import get_tokenizer
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.classification_common import QueryDocInstance, encode_query_doc_instance
from tlm.robust.load import load_robust_tokens_for_train, load_robust_tokens_for_predict
from trec.qrel_parse import load_qrels_structured


class RobustSeparateEncoder:
    def __init__(self, doc_max_length, query_type="title", neg_k=1000, pos_only=True):
        self.data = self.load_tokens()
        qrel_path = "/home/youngwookim/Downloads/rob04-desc/qrels.rob04.txt"
        self.judgement = load_qrels_structured(qrel_path)
        self.doc_max_length = doc_max_length
        self.queries = load_robust_04_query(query_type)
        self.tokenizer = get_tokenizer()
        self.galago_rank = load_bm25_best()
        self.neg_k = neg_k
        self.pos_only = pos_only

    def load_tokens(self):
        tokens_d = load_robust_tokens_for_train()
        tokens_d.update(load_robust_tokens_for_predict(4))
        return tokens_d

    def generate(self, query_list, data_id_manager) -> Iterator[QueryDocInstance]:
        neg_k = self.neg_k
        for query_id in query_list:
            if query_id not in self.judgement:
                continue

            qck_query = QCKQuery(query_id, "")
            judgement = self.judgement[query_id]
            query = self.queries[query_id]
            query_tokens = self.tokenizer.tokenize(query)

            ranked_list = self.galago_rank[query_id]
            ranked_list = ranked_list[:neg_k]

            target_docs = set(judgement.keys())
            target_docs.update([e.doc_id for e in ranked_list])
            print("Total of {} docs".format(len(target_docs)))

            for doc_id in target_docs:
                tokens = self.data[doc_id][:self.doc_max_length]
                label = 1 if doc_id in judgement and judgement[doc_id] > 0 else 0
                if self.pos_only and not label:
                    continue
                candidate = QCKCandidate(doc_id, "")
                info = {
                    'query': get_light_qckquery(qck_query),
                    'candidate': get_light_qckcandidate(candidate),
                    'q_term_len': len(query_tokens),
                }
                data_id = data_id_manager.assign(info)
                inst = QueryDocInstance(query_tokens, tokens, label, data_id)
                yield inst

    def write(self, insts: Iterable[QueryDocInstance], out_path: str):
        def encode_fn(inst: QueryDocInstance) -> collections.OrderedDict :
            return encode_query_doc_instance(self.tokenizer, self.doc_max_length, inst)
        write_records_w_encode_fn(out_path, encode_fn, insts, 0)
