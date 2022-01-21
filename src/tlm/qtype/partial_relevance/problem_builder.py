from typing import Dict, List, Tuple

from arg.qck.decl import get_format_handler
from arg.qck.prediction_reader import load_combine_info_jsons
from data_generator.bert_input_splitter import split_p_h_with_input_ids
from data_generator.tokenizer_wo_tf import is_continuation, get_tokenizer
from scipy_aux import logit_to_score_softmax
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer
from tlm.qtype.content_functional_parsing.qid_to_content_tokens import QueryInfo, load_query_info_dict
from tlm.qtype.partial_relevance.eval_data_structure import SegmentedInstance, RelatedEvalInstance, SegmentedText


def get_word_level_location_w_ids(tokenizer, input_ids) -> List[List[int]]:
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    intervals: List[List[int]] = []
    start = 0
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if idx == 0:
            pass
        elif is_continuation(token):
            pass
        elif token == "[PAD]":
            break
        else:
            end = idx
            intervals.append(list(range(start, end)))
            start = idx
        idx += 1
    end = idx
    if end > start:
        l: List[int] = list(range(start, end))
        intervals.append(l)
    return intervals


def word_segment_w_indices(tokenizer, input_ids) -> Tuple[List[int], List[List[int]]]:
    word_location_list = get_word_level_location_w_ids(tokenizer, input_ids)
    return input_ids, word_location_list


def build_eval_instances(info_path,
                         raw_prediction_path,
                         segment_text2_fn,
                         n_item=None) -> List[RelatedEvalInstance]:
    query_info_dict: Dict[str, QueryInfo] = load_query_info_dict("dev")
    f_handler = get_format_handler("qc")
    data_info: Dict = load_combine_info_jsons(info_path, f_handler.get_mapping(), f_handler.drop_kdp())
    viewer = EstimatorPredictionViewer(raw_prediction_path)
    tokenizer = get_tokenizer()
    all_items = []
    for idx, e in enumerate(viewer):
        if n_item is not None and idx >= n_item:
            break

        data_id = e.get_vector("data_id")[0]
        input_ids = e.get_vector("input_ids")
        logits = e.get_vector("logits")
        info_entry = data_info[str(data_id)]
        query_id = info_entry['query'].query_id
        doc_id = info_entry['candidate'].id
        query_tokens = info_entry['query'].text
        q_tokens_ids, d_tokens_ids = split_p_h_with_input_ids(input_ids, input_ids)
        score = logit_to_score_softmax(logits)
        d_tokens_ids = d_tokens_ids.tolist()
        d_tokens_ids, seg2_indices = segment_text2_fn(tokenizer, d_tokens_ids)
        query_info = query_info_dict[query_id]
        q_seg_indices = query_info.get_q_seg_indices()
        text1 = SegmentedText(q_tokens_ids.tolist(), q_seg_indices)
        text2 = SegmentedText(d_tokens_ids, seg2_indices)
        si = SegmentedInstance(text1, text2)
        problem_id = "{}-{}".format(query_id, doc_id)
        rei = RelatedEvalInstance(problem_id, query_info, si, score.tolist())
        assert 1 >= score >= 0
        all_items.append(rei)
    return all_items