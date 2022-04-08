from typing import Dict, List

from arg.qck.decl import get_format_handler
from arg.qck.prediction_reader import load_combine_info_jsons
from bert_api.segmented_instance.seg_instance import SegmentedInstance
from bert_api.segmented_instance.segmented_text import SegmentedText
from data_generator.bert_input_splitter import split_p_h_with_input_ids
from data_generator.tokenizer_wo_tf import get_tokenizer
from scipy_aux import logit_to_score_softmax
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer
from tlm.qtype.content_functional_parsing.qid_to_content_tokens import QueryInfo, load_query_info_dict
from contradiction.alignment.data_structure.related_eval_instance import RelatedEvalInstance


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


def build_sentence_as_doc(info_path,
                         raw_prediction_path,
                         sentence_segment_fn,
                         word_segment_fn,
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
        info_entry = data_info[str(data_id)]
        query_id = info_entry['query'].query_id
        doc_id = info_entry['candidate'].id
        q_tokens_ids, d_tokens_ids = split_p_h_with_input_ids(input_ids, input_ids)
        d_tokens_ids = d_tokens_ids.tolist()
        d_tokens_ids, sents_indices = sentence_segment_fn(tokenizer, d_tokens_ids)
        query_info = query_info_dict[query_id]
        q_seg_indices = query_info.get_q_seg_indices()
        text1 = SegmentedText(q_tokens_ids.tolist(), q_seg_indices)

        for sent_idx, sent_indices in enumerate(sents_indices):
            new_d_tokens_ids = [d_tokens_ids[i] for i in sent_indices]
            new_d_tokens_ids, seg_indices = word_segment_fn(tokenizer, new_d_tokens_ids)
            text2 = SegmentedText(new_d_tokens_ids, seg_indices)
            si = SegmentedInstance(text1, text2)
            problem_id = "{}-{}-{}".format(query_id, doc_id, sent_idx)
            dummy_score = 0
            rei = RelatedEvalInstance(problem_id, query_info, si, dummy_score)
            all_items.append(rei)
    return all_items

