from arg.perspectives.contextual_hint_analysis import predict_with_lm
from arg.perspectives.evaluate import evaluate
from arg.perspectives.load import load_dev_claim_ids, get_claims_from_ids


def run_baseline():
    d_ids = list(load_dev_claim_ids())
    claims = get_claims_from_ids(d_ids)
    top_k = 20
    #pred = predict_by_elastic_search(claims, top_k)
    pred = predict_with_lm(claims, top_k)
    print(evaluate(pred))


if __name__ == "__main__":
    run_baseline()
