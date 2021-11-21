from typing import Dict, Tuple, List, NamedTuple

import tlm.qtype.qe_de_res_parse
from arg.qck.decl import get_format_handler
from arg.qck.prediction_reader import load_combine_info_jsons
from cpath import at_output_dir
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import find_where
from misc_lib import two_digit_float, str_float_list
from tlm.data_gen.robust_gen.seg_lib.seg_score_common import ScoredPieceFromPair, OutputViewer, get_probs, \
    get_piece_scores


class PieceScoreParser:
    def __init__(self, queries, qid_list, probe_config):
        self.long_seg_score_path_format = at_output_dir("rqd", "rqd_{}.score")
        self.short_seg_score_path_format = at_output_dir("rqd", "rqd_sm_{}.score")
        info_file_path = at_output_dir("robust", "seg_info")
        f_handler = get_format_handler("qc")
        self.f_handler = f_handler
        self.info: Dict = load_combine_info_jsons(info_file_path, f_handler.get_mapping(), f_handler.drop_kdp())
        self.doc_piece_score_d: Dict[Tuple[str, str], List[ScoredPieceFromPair]] = {}
        self.prepared_qids = set()
        self.probe_config = probe_config
        self.queries = queries
        self.tokenizer = get_tokenizer()
        self.qid_list: List[str] = qid_list
        self.not_found_cnt = 0

    def get_qterm_len(self, qid) -> int:
        query = self.queries[str(qid)]
        q_term_length = len(self.tokenizer.tokenize(query))
        return q_term_length

    def prepare_for_query_id(self, query_id: str):
        job_id = self.qid_list.index(query_id)
        n_factor = self.probe_config.n_factor
        batch_size = self.probe_config.batch_size
        max_seq_length = self.probe_config.max_seq_length
        max_seq_length2 = self.probe_config.max_seq_length2
        step_size = self.probe_config.step_size
        long_seg_score_path = self.long_seg_score_path_format.format(job_id)
        short_seg_score_path = self.short_seg_score_path_format.format(job_id)

        qid = self.qid_list[job_id]
        q_term_length = self.get_qterm_len(qid)
        data1 = OutputViewer(long_seg_score_path, n_factor, batch_size)
        data2 = OutputViewer(short_seg_score_path, n_factor, batch_size)
        segment_len = max_seq_length - 3 - q_term_length
        segment_len2 = max_seq_length2 - 3 - q_term_length

        doc_piece_score_d: Dict[Tuple[str, str], List[ScoredPieceFromPair]] = {}
        for d1, d2 in zip(data1, data2):
            # for each query, doc pairs
            cur_info1 = self.info[d1['data_id']]
            cur_info2 = self.info[d2['data_id']]
            query_doc_id1 = tlm.qtype.qe_de_res_parse.get_pair_id(cur_info1)
            query_doc_id2 = tlm.qtype.qe_de_res_parse.get_pair_id(cur_info2)

            assert query_doc_id1 == query_doc_id2

            doc = d1['doc']
            probs = get_probs(d1['logits'])
            probs2 = get_probs(d2['logits'])
            piece_scores: List[Tuple[int, int, float]] = get_piece_scores(n_factor, probs, segment_len, step_size)
            piece_scores2: List[Tuple[int, int, float]] = get_piece_scores(n_factor, probs2, segment_len2, step_size)
            ss_list = []
            for st, ed, score in piece_scores:
                try:
                    st2, ed2, score2 = find_where(lambda x: x[1] == ed, piece_scores2)
                    assert ed == ed2
                    assert st < st2
                    tokens = self.tokenizer.convert_ids_to_tokens(doc[st: st2])
                    diff = score - score2
                    ss = ScoredPieceFromPair(st, st2, diff, score, score2, tokens)
                    ss_list.append(ss)
                except StopIteration:
                    pass
            doc_piece_score_d[query_doc_id1] = ss_list

        print("{} query-doc score is loaded".format(len(doc_piece_score_d)))
        self.doc_piece_score_d.update(doc_piece_score_d)
        self.prepared_qids.add(query_id)

    def get_piece_score(self, query_id, doc_id) -> List[ScoredPieceFromPair]:
        if query_id not in self.prepared_qids:
            self.prepare_for_query_id(query_id)

        try:
            return self.doc_piece_score_d[(query_id, doc_id)]
        except KeyError as e:
            self.not_found_cnt += 1
            if self.not_found_cnt % 10 == 0:
                print("{} doc not found so far".format(self.not_found_cnt))
            raise e

    def has_piece_score(self, query_id, doc_id) -> bool:
        if query_id not in self.prepared_qids:
            self.prepare_for_query_id(query_id)
        return (query_id, doc_id) in self.doc_piece_score_d


class Piece(NamedTuple):
    st: int
    ed: int


class ScoredInterval(NamedTuple):
    st: int
    ed: int
    piece_scores: List[float]
    location_score: float
    score: float

    def __str__(self):
        return str((self.st, self.ed)) \
               + " " + str_float_list(self.piece_scores) \
               + " " + two_digit_float(self.location_score)


class PiecewiseSegment(NamedTuple):
    piece_list: List[ScoredInterval]
    score: float

    def __str__(self):
        text = ""
        for piece in self.piece_list:
            text += str(piece) + " "
        return text

