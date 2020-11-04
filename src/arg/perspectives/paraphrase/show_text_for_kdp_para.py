import sys

from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import load_claim_ids_for_split, get_claims_from_ids, claims_to_dict
from data_generator.tokenizer_wo_tf import pretty_tokens
from datastore.interface import load
from datastore.table_names import BertTokenizedCluewebDoc


def main():
    f = open(sys.argv[1], "r")
    ids = load_claim_ids_for_split("train")
    claims = get_claims_from_ids(ids)
    claims_d = claims_to_dict(claims)
    ##
    for line in f:
        claim_id, p_id, doc_id, sent_idx = line.split()
        print("Claim {}\t{}".format(claim_id, claims_d[int(claim_id)]))
        print("Pers {}\t{}".format(p_id, perspective_getter(int(p_id))))
        doc = load(BertTokenizedCluewebDoc, doc_id)
        sent = doc[int(sent_idx)]
        s = pretty_tokens(sent, True)
        print("doc {} - {} : {}".format(doc_id, sent_idx, s))



if __name__ == "__main__":
    main()