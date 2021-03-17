import collections
from typing import Iterator, Iterable

from arg.qck.decl import QCKQuery, QCKCandidate, get_light_qckquery, get_light_qckcandidate
from data_generator.data_parser.robust import load_robust_04_query
from data_generator.data_parser.robust2 import load_bm25_best
from data_generator.tokenizer_wo_tf import get_tokenizer
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.classification_common import ClassificationInstanceWDataID, encode_classification_instance_w_data_id
from tlm.robust.load import load_robust_tokens_for_train, load_robust_tokens_for_predict
from trec.qrel_parse import load_qrels_structured


class RobustPosOnlyGen:
    def __init__(self, encoder, max_seq_length, query_type="title", neg_k=1000):
        self.data = self.load_tokens()
        qrel_path = "/home/youngwookim/Downloads/rob04-desc/qrels.rob04.txt"
        self.judgement = load_qrels_structured(qrel_path)
        self.max_seq_length = max_seq_length
        self.queries = load_robust_04_query(query_type)
        self.encoder = encoder
        self.tokenizer = get_tokenizer()
        self.galago_rank = load_bm25_best()
        self.neg_k = neg_k

    def load_tokens(self):
        tokens_d = load_robust_tokens_for_train()
        tokens_d.update(load_robust_tokens_for_predict(4))
        return tokens_d

    def generate(self, query_list, data_id_manager) -> Iterator[ClassificationInstanceWDataID]:
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
                tokens = self.data[doc_id]
                passage_list = self.encoder.encode(query_tokens, tokens)
                label = 1 if doc_id in judgement and judgement[doc_id] > 0 else 0
                if not label:
                    continue
                candidate = QCKCandidate(doc_id, "")
                for idx, (tokens, seg_ids) in enumerate(passage_list):
                    info = {
                        'query': get_light_qckquery(qck_query),
                        'candidate': get_light_qckcandidate(candidate),
                        'idx': idx,
                    }
                    data_id = data_id_manager.assign(info)
                    inst = ClassificationInstanceWDataID(tokens, seg_ids, label, data_id)
                    yield inst

    def write(self, insts: Iterable[ClassificationInstanceWDataID], out_path: str):
        def encode_fn(inst: ClassificationInstanceWDataID) -> collections.OrderedDict :
            return encode_classification_instance_w_data_id(self.tokenizer, self.max_seq_length, inst)
        write_records_w_encode_fn(out_path, encode_fn, insts, 0)
