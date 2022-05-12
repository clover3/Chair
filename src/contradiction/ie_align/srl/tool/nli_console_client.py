from bert_api import SegmentedText
from bert_api.segmented_instance.segmented_text import get_word_level_segmented_text_from_str
from bert_api.task_clients.nli_interface.nli_interface import NLIInput
from bert_api.task_clients.nli_interface.nli_predictors import get_nli_client_by_server
from contradiction.medical_claims.token_tagging.visualizer.deletion_score_to_html import make_nli_prediction_summary_str
from data_generator.tokenizer_wo_tf import get_tokenizer


def main():
    predict = get_nli_client_by_server()
    tokenizer = get_tokenizer()
    while True:
        sent1 = input("Premise: ")
        sent2 = input("Hypothesis: ")
        t_text1: SegmentedText = get_word_level_segmented_text_from_str(tokenizer, sent1)
        t_text2: SegmentedText = get_word_level_segmented_text_from_str(tokenizer, sent2)

        probs = predict([NLIInput(t_text1, t_text2)])[0]
        pred_summary = make_nli_prediction_summary_str(probs)
        print((sent1, sent2))
        print(pred_summary)


if __name__ == "__main__":
    main()
