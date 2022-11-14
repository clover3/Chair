import scipy.special

from bert_api.segmented_instance.segmented_text import SegmentedText, get_word_level_segmented_text_from_str, \
    seg_to_text
from bert_api.task_clients.nli_interface.nli_interface import NLIInput, NLIPredictorFromSegTextSig
from bert_api.task_clients.nli_interface.nli_predictors import get_nli_client
from contradiction.medical_claims.token_tagging.intersection_search.tree_deletion_search import TreeDeletionSearch
from contradiction.medical_claims.token_tagging.problem_loader import load_alamri_split
from data_generator.NLI.enlidef import nli_probs_str
from data_generator.tokenizer_wo_tf import get_tokenizer


def get_neutral_probability(probs):
    return probs[1]


def main():
    problem_list = load_alamri_split("dev")[:2]
    tokenizer = get_tokenizer()
    # cache_client = get_nli_cache_client("localhost")
    forward_fn_raw: NLIPredictorFromSegTextSig = get_nli_client("localhost")
    predict_fn = forward_fn_raw
    sds = TreeDeletionSearch(predict_fn)
    for p in problem_list:
        t_text1: SegmentedText = get_word_level_segmented_text_from_str(tokenizer, p.text1)
        t_text2: SegmentedText = get_word_level_segmented_text_from_str(tokenizer, p.text2)
        common: SegmentedText = sds.find_intersection(t_text1, t_text2)
        print("text1: ", seg_to_text(tokenizer, t_text1))
        print("text2: ", seg_to_text(tokenizer, t_text2))
        print("common:", seg_to_text(tokenizer, common))
        def get_decision_res(t_text):
            logits = predict_fn([NLIInput(t_text, common)])[0]
            probs = scipy.special.softmax(logits)
            return nli_probs_str(probs)

        print("text1:common: ", get_decision_res(t_text1))
        print("text2:common: ", get_decision_res(t_text2))


if __name__ == "__main__":
    main()
