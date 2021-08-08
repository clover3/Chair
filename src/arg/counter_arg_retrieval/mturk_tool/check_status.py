from collections import Counter

import boto3


def get_client():
    active_url = 'https://mturk-requester.us-east-1.amazonaws.com'
    mturk = boto3.client('mturk',
                         region_name='us-east-1',
                         endpoint_url=active_url)
    return mturk

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
            if hit['HITTypeId'] == "3QUMZFVHEAQ4IBL8VX8HJMK1TRDR6P":
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
        # print(hit['HITId'], hit['MaxAssignments'], hit['NumberOfAssignmentsAvailable'],
        #       hit['NumberOfAssignmentsCompleted'],
        #       hit['NumberOfAssignmentsPending'])
        print(hit)

    for key, cnt in counter.items():
        print(key, cnt)
    print(counter_hit)



if __name__ == "__main__":
    view_hits()
