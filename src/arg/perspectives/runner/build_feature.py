from arg.perspectives.basic_analysis import get_candidates
from arg.perspectives.load import get_claims_from_ids, load_train_claim_ids


def work():
    d_ids = list(load_train_claim_ids())
    claims = get_claims_from_ids(d_ids)
    all_data_points = get_candidates(claims)


if __name__ == "__main__":
    NotImplemented
