from contradiction.ie_align.srl.tool.get_nli_predictor_common import get_nli_predictor
from contradiction.medical_claims.token_tagging.visualizer.deletion_score_to_html import make_prediction_summary_str


def main():
    predict = get_nli_predictor()
    while True:
        sent1 = input("Premise: ")
        sent2 = input("Hypothesis: ")

        tokens1 = sent1.split("\t")
        tokens2 = sent2.split("\t")

        for t1, t2 in zip(tokens1, tokens2):
            probs = predict(t1, t2)
            pred_summary = make_prediction_summary_str(probs)
            print("{} / {} / {}".format(pred_summary, t1, t2))




if __name__ == "__main__":
    main()