from collections import Counter

import boto3

from cache import save_to_pickle, load_from_pickle


def ask_hit_id():
    mturk = get_client()
    creation_time_key = "CreationTime"

    hits_with_type = []
    next_token = ""
    while True:
        if next_token:
            res = mturk.list_hits(MaxResults=100, NextToken=next_token)
        else:
            res = mturk.list_hits(MaxResults=100)
        print("{} hits ".format(len(res["HITs"])))
        for hit in res["HITs"]:
            if hit['HITTypeId'] == "34A9VSEXPFWA264TLZ2K15PC9BQ9JP":
                hits_with_type.append(hit)
                # print(hit['HITId'], hit[creation_time_key])
        if 'NextToken' in res:
            next_token = res['NextToken']
        else:
            break

    save_to_pickle(hits_with_type, "hits")



def view_hits():
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
            if hit['HITTypeId'] == "34A9VSEXPFWA264TLZ2K15PC9BQ9JP":
                hits_with_type.append(hit)
                # print(hit['HITId'], hit[creation_time_key])
        if 'NextToken' in res:
            next_token = res['NextToken']
        else:
            break
    print("{} hits ".format(len(hits_with_type)))

    counter_hit = Counter()
    counter = Counter()
    for hit in hits_with_type:
        print(hit['HITId'], hit['MaxAssignments'], hit['NumberOfAssignmentsAvailable'],
              hit['NumberOfAssignmentsCompleted'],
              hit['NumberOfAssignmentsPending'])


    for key, cnt in counter.items():
        print(key, cnt)
    print(counter_hit)


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


def get_client():
    active_url = 'https://mturk-requester.us-east-1.amazonaws.com'
    mturk = boto3.client('mturk',
                         region_name='us-east-1',
                         endpoint_url=active_url)
    return mturk


if __name__ == "__main__":
    main()