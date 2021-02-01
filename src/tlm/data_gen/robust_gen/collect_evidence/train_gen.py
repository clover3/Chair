import collections
from typing import List, Tuple

from numpy import argsort

from data_generator.data_parser.robust import load_robust_04_query
from data_generator.data_parser.robust2 import load_bm25_best
from data_generator.tokenizer_wo_tf import get_tokenizer
from evals.parse import load_qrels_structured
from misc_lib import ceil_divide, get_first
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.adhoc_datagen import get_combined_tokens_segment_ids, SegmentIDs, Tokens
from tlm.data_gen.classification_common import ClassificationInstance, encode_classification_instance
from tlm.robust.load import load_robust_tokens_for_train, load_robust_tokens_for_predict


class RobustCollectedEvidenceTrainGen:
    def __init__(self,
                 encoder,
                 max_seq_length,
                 query_type="title",
                 neg_k=1000):
        self.data = self.load_tokens()
        qrel_path = "/home/youngwookim/Downloads/rob04-desc/qrels.rob04.txt"
        self.judgement = load_qrels_structured(qrel_path)
        self.max_seq_length = max_seq_length
        self.queries = load_robust_04_query(query_type)
        self.encoder = encoder
        self.tokenizer = get_tokenizer()
        self.galago_rank = load_bm25_best()
        self.neg_k = neg_k
        self.n_seg_per_doc = 4

    def load_tokens(self):
        tokens_d = load_robust_tokens_for_train()
        tokens_d.update(load_robust_tokens_for_predict(4))
        return tokens_d

    def generate(self, query_list) -> List[ClassificationInstance]:
        neg_k = self.neg_k
        all_insts = []
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
            print("Total of {} docs".format(len(target_docs)))

            for doc_id in target_docs:
                insts: List[Tuple[List, List]] = self.encode(query_tokens, doc_id)
                tokens = self.data[doc_id]
                insts: List[Tuple[List, List]] = self.encoder.encode(query_tokens, tokens)
                label = 1 if doc_id in judgement and judgement[doc_id] > 0 else 0

                for tokens_seg, seg_ids in insts:
                    assert type(tokens_seg[0]) == str
                    assert type(seg_ids[0]) == int
                    all_insts.append(ClassificationInstance(tokens_seg, seg_ids, label))

        return all_insts

    # # # # # # # # # # # # #
    # Doc length : 4000 tokens
    # Input max_seq_length : 128
    # Line length : 16
    # Selection unit : (64) + (64 - n_q_tokens)
    # # # # # # # # # # # # #
    def encode(self, query_tokens, doc_id) -> List[Tuple[Tokens, SegmentIDs]]:
        all_tokens: List[str] = self.data[doc_id]
        score: List[float] = NotImplemented
        assert len(all_tokens) == len(score)
        window_size = self.max_seq_length - len(query_tokens) - 3
        idx_list: List[Tuple[int, int]] = select_window(score, window_size, self.n_seg_per_doc)

        output = []
        for st, ed in idx_list:
            content_tokens = all_tokens[st:ed][:window_size]
            tokens, segments_ids = get_combined_tokens_segment_ids(query_tokens, content_tokens)
            output.append((tokens, segments_ids))

        return output

    def write(self, insts: List[ClassificationInstance], out_path: str):
        def encode_fn(inst: ClassificationInstance) -> collections.OrderedDict :
            return encode_classification_instance(self.tokenizer, self.max_seq_length, inst)
        write_records_w_encode_fn(out_path, encode_fn, insts, len(insts))


def select_window(score: List[float], window_size, n_seg_per_doc) -> List[Tuple[int, int]]:
    assert len(score) >= window_size * n_seg_per_doc
    line_len = 16
    n_line = ceil_divide(len(score), line_len)
    line_scores: List[float] = list([sum(score[i*line_len: (i+1)*line_len]) for i in range(n_line)])
    idx_sorted = argsort(line_scores)[::-1]
    line_per_window = ceil_divide(window_size, line_len)
    best_idx = idx_sorted[0]
    selected_windows = []
    for _ in range(n_seg_per_doc):
        candidate_list = []
        for i in range(line_per_window):
            st = best_idx - i
            ed = best_idx - i + line_per_window
            score_sum = sum(line_scores[st:ed])
            e = (st, ed, score_sum)
            candidate_list.append(e)

        candidate_list.sort(key=lambda x: x[2], reverse=True)
        st, ed, _ = candidate_list[0]

        for j in range(st, ed):
            line_scores[j] = 0

        selected_windows.append((st, ed))

    selected_windows.sort(key=get_first)
    return selected_windows

