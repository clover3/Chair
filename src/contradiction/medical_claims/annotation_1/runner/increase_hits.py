from collections import Counter

from cache import save_to_pickle, load_from_pickle
from contradiction.medical_claims.mturk.mturk_api_common import get_client, ask_hit_id


def ask_save():
    target_hit_type_id = "34A9VSEXPFWA264TLZ2K15PC9BQ9JP"
    hits_with_type = ask_hit_id(target_hit_type_id)
    save_to_pickle(hits_with_type, "hits")


def view_hits():
    hit_type_id = "3MOP54T9IQ4I11BF73KUIH37JB40YX"
    hits_with_type = query_list_hits(hit_type_id)
    print("{} hits ".format(len(hits_with_type)))

    counter_hit = Counter()
    n_avail = 0
    for hit in hits_with_type:
        n_avail += hit['NumberOfAssignmentsAvailable']
        if hit['NumberOfAssignmentsAvailable']:
            print(hit['HITId'], hit['MaxAssignments'], hit['NumberOfAssignmentsAvailable'],
                  hit['NumberOfAssignmentsCompleted'],
                  hit['NumberOfAssignmentsPending'])

    print("Total of {} assignments available".format(n_avail))
    print(counter_hit)


def query_list_hits(hit_type_id):
    mturk = get_client()
    hits_with_type = []
    next_token = ""
    while True:
        if next_token:
            res = mturk.list_hits(MaxResults=100, NextToken=next_token)
        else:
            res = mturk.list_hits(MaxResults=100)
        print("{} hits ".format(len(res["HITs"])))
        for hit in res["HITs"]:
            if hit['HITTypeId'] == hit_type_id:
                hits_with_type.append(hit)
                # print(hit['HITId'], hit[creation_time_key])
        if 'NextToken' in res:
            next_token = res['NextToken']
        else:
            break
    return hits_with_type


def checkHits():
    hits_with_type = load_from_pickle("hits")
    counter = Counter()
    for hit in hits_with_type:
        key = hit["CreationTime"], hit['HITGroupId'], hit['Title'], hit['RequesterAnnotation']
        counter[key] += 1

    for key, cnt in counter.items():
        print(key, cnt)


def load_hit_id():
    hits_with_type = load_from_pickle("hits")
    return list([h['HITId'] for h in hits_with_type])


def main():
    view_hits()
    # add_hits(load_hit_id())


def add_hits(hit_id_list):
    mturk = get_client()
    for idx, hit_id in enumerate(hit_id_list[1:]):
        ret = mturk.create_additional_assignments_for_hit(HITId=hit_id, NumberOfAdditionalAssignments=2, UniqueRequestToken=hit_id)
        if ret['ResponseMetadata']['HTTPStatusCode'] == 200:
            continue
        else:
            print(hit_id, idx)
            break


def change_hit_type():
    hit_id_list = [
        "3R6RZGK0XG8E4B3CG832IU17YZGYVM",
        "3ZUE82NE0BXAT8Q43P041VAJHQQ8FV",
        "3IH9TRB0FCVCSZ895CXAPI03JCC1IV",
        "3OQQD2WO8J2822MOSGBTDBF0G643IM",
        "3D5G8J4N5B0INP4I62G3AD9SI7CTVP",
        "3RWB1RTQDKJMKLFYHSW9DPONI6R8PV",
        "3X2YVV51PV0UTUSEUT1PQ99BXXNW1T",
    ]
    mturk = get_client()
    new_hit_type_id = "3BHRGUFAIAS5Z7UMKDLPVWH7BFFCSY"
    for hit_id in hit_id_list:
        ret = mturk.update_hit_type_of_hit(HITId=hit_id, HITTypeId =new_hit_type_id)
        print(ret)



if __name__ == "__main__":
    checkHits()