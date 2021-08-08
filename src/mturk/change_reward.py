import boto3


def change_reward(hit_id_list, hit_type):
    mturk = get_client()
    res = mturk.list_hits()
    maybe_hits = res['HITs']
    for hit in maybe_hits:
        hit_id = hit['HITId']
        print(hit['HITId'], hit['HITTypeId'], hit['CreationTime'])

        response = mturk.update_hit_type_of_hit(
            HITId=hit_id, HITTypeId=hit_type
        )

        print(response)


def get_client():
    sandbox_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
    active_url = 'https://mturk-requester.us-east-1.amazonaws.com'
    mturk = boto3.client('mturk',
                         region_name='us-east-1',
                         endpoint_url=active_url)
    return mturk


def enum_hit_with_type(hit_type):
    mturk = get_client()
    res = mturk.list_hits()
    maybe_hits = res['HITs']
    output_hit = []
    for hit in maybe_hits:
        if hit['HITTypeId'] == hit_type:
            output_hit.append(hit)
    return output_hit


def main():
    original_type = "3RW2QOCBNK9RA4ZW31D76R1N34NOO2"
    new_type = "3QUMZFVHEAQ4IBL8VX8HJMK1TRDR6P"
    hits = enum_hit_with_type(original_type)
    hit_ids = []
    for h in hits:
        hit_ids.append(h['HITId'])
    print("{} hits found".format(len(hits)))
    change_reward(hit_ids, new_type)


if __name__ == "__main__":
    main()