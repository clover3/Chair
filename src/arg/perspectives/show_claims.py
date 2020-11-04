import sys

from arg.perspectives.load import load_claim_ids_for_split, get_claims_from_ids


def main():
    split = sys.argv[1]

    ids = load_claim_ids_for_split(split)
    claims = get_claims_from_ids(ids)

    for c in claims:
        print("Claim {} :\t{}".format(c['cId'], c['text']))


if __name__ == "__main__":
    main()