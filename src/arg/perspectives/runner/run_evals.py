from arg.perspectives.collection_based_classifier import predict_by_mention_num
from arg.perspectives.evaluate import evaluate
from arg.perspectives.load import get_claims_from_ids, load_train_claim_ids
from misc_lib import split_7_3


def run_baseline():
    d_ids = list(load_train_claim_ids())
    claims = get_claims_from_ids(d_ids)
    train, val = split_7_3(claims)
    top_k = 5
    #pred = predict_by_elastic_search(claims, top_k)
    pred = predict_by_mention_num(val, top_k)
    #pred = predict_with_lm(val, top_k)
    print(evaluate(pred))


if __name__ == "__main__":
    run_baseline()
