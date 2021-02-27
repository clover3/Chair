import collections
from queue import Queue
from typing import Tuple, NamedTuple

from data_generator.data_parser.robust import load_robust_04_query
from data_generator.data_parser.robust2 import load_bm25_best
from data_generator.tokenizer_wo_tf import get_tokenizer
from evals.parse import load_qrels_structured
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.classification_common import ClassificationInstance
from tlm.data_gen.robust_gen.select_supervision.score_selection_methods import *
from tlm.robust.load import load_robust_tokens_for_train, load_robust_tokens_for_predict


class SegDoc(NamedTuple):
    tokens: List[str]
    seg_ids: List[int]


class DocInst(NamedTuple):
    segs: List[SegDoc]
    label: int


class RobustVarLengthDocTrainGen:
    def __init__(self, encoder,
                 max_seq_length_per_inst,
                 num_doc_per_inst,
                 num_seg_per_inst,
                 query_type="title", neg_k=1000):
        self.data = self.load_tokens()
        qrel_path = "/home/youngwookim/Downloads/rob04-desc/qrels.rob04.txt"
        self.judgement = load_qrels_structured(qrel_path)
        self.max_seq_length = max_seq_length_per_inst
        self.queries = load_robust_04_query(query_type)
        self.num_doc_per_inst = num_doc_per_inst
        self.num_seg_per_inst = num_seg_per_inst

        self.all_segment_encoder = encoder
        self.tokenizer = get_tokenizer()
        self.galago_rank = load_bm25_best()
        self.neg_k = neg_k

    def load_tokens(self):
        tokens_d = load_robust_tokens_for_train()
        tokens_d.update(load_robust_tokens_for_predict(4))
        return tokens_d

    def generate(self, query_list) -> List[List[DocInst]]:
        neg_k = self.neg_k
        long_doc_queue = Queue()
        short_doc_queue = Queue()

        long_docs = []
        short_docs = []
        for query_id in query_list:
            if query_id not in self.judgement:
                continue

            judgement = self.judgement[query_id]
            query = self.queries[query_id]
            query_tokens = self.tokenizer.tokenize(query)
            ranked_list = self.galago_rank[query_id]
            ranked_list = ranked_list[:neg_k]

            target_docs = set(judgement.keys())
            target_docs.update([e.doc_id for e in ranked_list])
            for doc_id in target_docs:
                tokens = self.data[doc_id]
                raw_segs: List[Tuple[List, List]] = self.all_segment_encoder.encode(query_tokens, tokens)

                segs = list([SegDoc(tokens, seg_ids) for tokens, seg_ids in raw_segs])
                label = 1 if doc_id in judgement and judgement[doc_id] > 0 else 0
                segs = segs[:self.num_seg_per_inst]
                per_doc = DocInst(segs, label)
                if len(segs) > 2:
                    long_doc_queue.put(per_doc)
                else:
                    short_doc_queue.put(per_doc)

        print("Short/Long : {}/{}".format(len(short_docs), len(long_docs)))

        def doc_remains():
            return not long_doc_queue.empty() or not short_doc_queue.empty()

        all_insts: List[List[DocInst]] = []
        while doc_remains():
            cur_docs = []
            n_segs = 0
            if not long_doc_queue.empty():
                doc_inst = long_doc_queue.get()
                cur_docs.append(doc_inst)
                n_segs += len(doc_inst.segs)

            while n_segs < self.num_seg_per_inst \
                    and len(cur_docs) < self.num_doc_per_inst \
                    and not short_doc_queue.empty():
                doc_inst = short_doc_queue.get()
                if n_segs + len(doc_inst.segs) <= self.num_seg_per_inst:
                    cur_docs.append(doc_inst)
                    n_segs += len(doc_inst.segs)
                else:
                    short_doc_queue.put(doc_inst)
                    break

            assert cur_docs
            all_insts.append(cur_docs)

        return all_insts

    def write(self, insts: List[SegDoc], out_path: str):
        def encode_fn(inst: ClassificationInstance) -> collections.OrderedDict :
            return NotImplemented
        write_records_w_encode_fn(out_path, encode_fn, insts, len(insts))
