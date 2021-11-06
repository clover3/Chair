from contradiction.ie_align.srl.tool.get_nli_predictor_common import get_nli_predictor
from contradiction.medical_claims.token_tagging.visualizer.deletion_score_to_html import make_prediction_summary_str


def main():
    predict = get_nli_predictor()
    while True:
        sent1 = input("Premise: ")
        sent2 = input("Hypothesis: ")
        probs = predict(sent1, sent2)
        pred_summary = make_prediction_summary_str(probs)
        print((sent1, sent2))
        print(pred_summary)


if __name__ == "__main__":
    main()
