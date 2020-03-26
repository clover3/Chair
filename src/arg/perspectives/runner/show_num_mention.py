from arg.perspectives.collection_based.datapoint_strct import get_num_mention
from arg.perspectives.collection_based_classifier import load_feature_and_split
from arg.perspectives.load import get_perspective_dict, get_claims_from_ids, claims_to_dict
from list_lib import lmap
from misc_lib import group_by


def show_num_mention():
    train, val = load_feature_and_split()
    p_dict = get_perspective_dict()
    claims = get_claims_from_ids(lmap(lambda x: x['cid'], train))
    claim_d = claims_to_dict(claims)
    grouped = group_by(train, lambda x: x['cid'])

    for cid in grouped:
        print("Claim:", claim_d[cid])
        for dp in grouped[cid]:
            p_text = p_dict[dp['pid']]
            print(dp['label'], get_num_mention(dp), p_text)



if __name__ =="__main__" :
    show_num_mention()
