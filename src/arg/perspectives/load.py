import json
import os
from collections import defaultdict
from typing import Iterable, List, Dict

from arg.perspectives.types import CPIDPair
from cpath import data_path
from list_lib import flatten
from misc_lib import split_7_3

dir_path = os.path.join(data_path, "perspective")

splits = ["train", "dev", "test"]

def load_json_by_name(name):
    return json.load(open(os.path.join(dir_path, name),"r"))


def load_topics():
    return load_json_by_name("topic.json")


def load_data_set_split():
    return load_json_by_name("dataset_split_v1.0.json")


# List[ {"pId": ~~, "text":~~, "source":~~~ } ]
def load_perspective_pool():
    return load_json_by_name("perspective_pool_v1.0.json")


def get_perspective_dict():
    d = {}
    for e in load_perspective_pool():
        pId = e['pId']
        text = e['text']
        d[pId] = text
    return d


# Load gold (claim, perspectives) list
def load_claim_perspective_pair():
    return load_json_by_name("perspectrum_with_answers_v1.0.json")


def load_train_claim_ids() -> Iterable[int]:
    return load_claim_ids_for_split('train')


def load_dev_claim_ids() -> Iterable[int]:
    return load_claim_ids_for_split('dev')


def load_test_claim_ids() -> Iterable[int]:
    return load_claim_ids_for_split('test')


def load_claim_ids_for_split(split) -> Iterable[int]:
    d = load_data_set_split()
    for c_id in d:
        if d[c_id] == split:
            yield int(c_id)


def load_claims_for_sub_split(sub_split) -> List[Dict]:
    if sub_split in ["train" , "val"]:
        split = "train"
        d_ids: List[int] = list(load_train_claim_ids())
        claims = get_claims_from_ids(d_ids)
        train, val = split_7_3(claims)
        if sub_split == "train":
            return train
        elif sub_split == "val":
            return val
        else:
            assert False

    else:
        split = sub_split
        ids = load_claim_ids_for_split(split)
        return get_claims_from_ids(ids)


d_n_claims_per_split = {
        'train': 378,
        'val': 162,
        'dev': 138
    }


def claims_to_dict(claims) -> Dict[int, str]:
    d = {}
    for e in claims:
        d[e['cId']] = e['text']
    return d



# get claim_per
def get_claim_perspective_id_dict() -> Dict[int, List[List[int]]]:
    claim_and_perspective = load_claim_perspective_pair()
    d = {}
    for e in claim_and_perspective:
        l: List[List[int]] = []
        for perspective_cluster in e['perspectives']:
            pids: List[int] = perspective_cluster['pids']
            l.append(pids)

        # n_unique_pers = len(set(flatten(l)))
        # n_total_pers = sum([len(s) for s in l])
        # if n_total_pers != n_unique_pers:
        #     print(n_total_pers, n_unique_pers)
        d[int(e['cId'])] = l
    return d


def get_claim_perspective_label_dict() -> Dict[CPIDPair, int]:
    gold = get_claim_perspective_id_dict()
    d = defaultdict(int)
    for cid, pid_list_list in gold.items():
        for pid in flatten(pid_list_list):
            cpid_pair = CPIDPair((cid, pid))
            d[cpid_pair] = 1
    return d


def get_claims_from_ids(claim_ids) -> List[Dict]:
    claim_ids_set = set(claim_ids)
    claim_and_perspective = load_claim_perspective_pair()
    output = []
    for e in claim_and_perspective:
        if e['cId'] in claim_ids_set:
            output.append(e)
    return output


def show_claim_perspective_pair():
    perspective = get_perspective_dict()
    claim_and_perspective = load_claim_perspective_pair()
    print(len(claim_and_perspective))

    for e in claim_and_perspective:
        claim_text = e['text']
        cid = e['cId']
        print("Claim {}: {} ".format(cid, claim_text))
        for perspective_cluster in e['perspectives']:
            pids = perspective_cluster['pids']
            stance_coarse = perspective_cluster['stance_label_3']
            print("Stance: ", stance_coarse)
            for pid in pids:
                print("P: ", perspective[pid])




if __name__ == "__main__":
    show_claim_perspective_pair()


