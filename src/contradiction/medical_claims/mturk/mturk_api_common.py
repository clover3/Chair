import boto3


def get_client():
    active_url = 'https://mturk-requester.us-east-1.amazonaws.com'
    mturk = boto3.client('mturk',
                         region_name='us-east-1',
                         endpoint_url=active_url)
    return mturk


def ask_hit_id(target_hit_type_id):
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
            if hit['HITTypeId'] == target_hit_type_id:
                hits_with_type.append(hit)
        if 'NextToken' in res:
            next_token = res['NextToken']
        else:
            break
    return hits_with_type



def get_all_available():
    mturk = get_client()
    out_hits = []
    next_token = ""
    while True:
        if next_token:
            res = mturk.list_hits(MaxResults=100, NextToken=next_token)
        else:
            res = mturk.list_hits(MaxResults=100)
        print("{} hits ".format(len(res["HITs"])))
        for hit in res["HITs"]:
            if hit['NumberOfAssignmentsAvailable']:
                out_hits.append(hit)
                # print(hit['HITId'], hit[creation_time_key])
        if 'NextToken' in res:
            next_token = res['NextToken']
        else:
            break
    print("{} hits ".format(len(out_hits)))
    return out_hits