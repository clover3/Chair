from collections import OrderedDict
from typing import Dict, List, Tuple, Iterable, NamedTuple

import scipy.special

from arg.perspectives.ppnc.parse_cpnr_results import get_recover_subtokens
from arg.qck.decl import QCKQuery, QCKCandidate, KDP, PayloadAsTokens
from arg.qck.encode_common import encode_two_inputs
from base_type import FilePath
from data_generator.tokenizer_wo_tf import CachedTokenizer, get_tokenizer
from estimator_helper.output_reader import join_prediction_with_info
from list_lib import lfilter, lmap
from misc_lib import group_by, DataIDManager
from tf_util.record_writer_wrap import write_records_w_encode_fn

recover_subtokens = NotImplemented


class QKOutEntry(NamedTuple):
    logits: List[float]
    query: QCKQuery
    kdp: KDP
    passage_tokens: List[str]

    @classmethod
    def from_dict(cls, d):
        return QKOutEntry(d['logits'], d['query'], d['kdp'],
                          recover_subtokens(d["input_ids"])
                          )


def collect_good_passages(data_id_to_info: Dict[str, Dict],
                          passage_score_path: FilePath,
                          config: Dict
                          ) -> List[Tuple[str, List[QKOutEntry]]]:
    global recover_subtokens
    recover_subtokens = get_recover_subtokens()
    score_cut = config['score_cut']
    top_k = config['top_k']
    score_type = config['score_type']
    fetch_field_list = ["logits", "input_ids"]
    data: List[Dict] = join_prediction_with_info(passage_score_path,
                                                 data_id_to_info,
                                                 fetch_field_list
                                                 )
    qk_out_entries: List[QKOutEntry] = lmap(QKOutEntry.from_dict, data)

    grouped: Dict[str, List[QKOutEntry]] = group_by(qk_out_entries, lambda x: x.query.query_id)


    def get_score_from_logit(logits) -> float:
        if score_type == "softmax":
            return scipy.special.softmax(logits)[1]
        elif score_type == "regression":
            return logits[0]
        else:
            assert False

    def get_score(entry: QKOutEntry):
        return get_score_from_logit(entry.logits)

    def is_good(qk_out_entry: QKOutEntry):
        score = get_score_from_logit(qk_out_entry.logits)
        return score >= score_cut

    output = []
    num_passges = []
    for cid, passages in grouped.items():
        good_passages = lfilter(is_good, passages)
        good_passages.sort(key=get_score, reverse=True)
        num_passges.append(len(good_passages))
        if good_passages:
            output.append((cid, good_passages[:top_k]))
        else:
            scores = lmap(get_score, passages)
            scores.sort(reverse=True)

    print(num_passges)
    print("{} of {} query has passages".format(len(output), len(grouped)))
    return output


def join_candidate(qk_output: List[Tuple[str, List[QKOutEntry]]],
                   candidate_id_dict: Dict[str, List[str]])\
        -> Iterable[Tuple[str, str, List[QKOutEntry]]]:

    for qid, passages in qk_output:
        for candidate_id in candidate_id_dict[qid]:
            e: Tuple[str, str, List[QKOutEntry]] = qid, candidate_id, passages
            yield e


QCKCompactEntry = Tuple[QCKQuery, QCKCandidate, QKOutEntry]


def add_query_and_candidate_text(joined_payloads: Iterable[Tuple[str, str, List[QKOutEntry]]] ,
                                 query_dict: Dict[str, QCKQuery],
                                 candidate_dict: Dict[str, QCKCandidate],
                                 ) -> Iterable[QCKCompactEntry]:

    for query_id, candidate_id, kdp_list in joined_payloads:
        query = query_dict[query_id]
        candidate = candidate_dict[candidate_id]
        for kdp in kdp_list:
            yield query, candidate, kdp


def write_qck_as_tfrecord(save_path, payloads: Iterable[QCKCompactEntry]):
    data_id_man = DataIDManager(0, 1000 * 1000)

    tokenizer = get_tokenizer()
    cache_tokenizer = CachedTokenizer(tokenizer)
    max_seq_length = 512

    def encode_fn(e: QCKCompactEntry) -> OrderedDict:
        query, candidate, qk_out_entry = e
        candidate: QCKCandidate = candidate
        info = {
            'query': query,
            'candidate': candidate,
            'kdp': qk_out_entry.kdp
        }

        p = PayloadAsTokens(passage=qk_out_entry.passage_tokens,
                            text1=cache_tokenizer.tokenize(query.text),
                            text2=cache_tokenizer.tokenize(candidate.text),
                            data_id=data_id_man.assign(info),
                            is_correct=0
                            )
        return encode_two_inputs(max_seq_length, tokenizer, p)

    write_records_w_encode_fn(save_path, encode_fn, payloads)
    return data_id_man


def qck_from_qk_results(qk_result: List[Tuple[str, List[QKOutEntry]]],
                        query_id_to_candidate_id_dict: Dict[str, List[str]],
                        query_dict: Dict[str, QCKQuery],
                        candidate_dict: Dict[str, QCKCandidate],
                        ) -> Iterable[QCKCompactEntry]:
    joined_payloads: Iterable[Tuple[str, str, List[QKOutEntry]]] = list(join_candidate(qk_result, query_id_to_candidate_id_dict))

    payloads: Iterable[QCKCompactEntry] = add_query_and_candidate_text(joined_payloads, query_dict, candidate_dict)
    return payloads





