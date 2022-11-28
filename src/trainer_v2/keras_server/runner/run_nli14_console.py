from contradiction.medical_claims.token_tagging.visualizer.deletion_score_to_html import make_nli_prediction_summary_str
from trainer_v2.keras_server.name_short_cuts import get_nli14_predictor


def main():
    predict = get_nli14_predictor()
    while True:
        sent1 = input("Premise: ")
        sent2 = input("Hypothesis: ")
        probs = predict([(sent1, sent2)])[0]
        pred_summary = make_nli_prediction_summary_str(probs)
        print((sent1, sent2))
        print(pred_summary)


if __name__ == "__main__":
    main()
