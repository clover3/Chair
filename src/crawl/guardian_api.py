import json

import requests


def get_comment(short_id):
    url_prefix = "http://discussion.guardianapis.com/discussion-api/discussion/"

    url = url_prefix + short_id
    print(url, end=" ")
    res = requests.get(url)
    if res.status_code == 200 :
        print("success ")
        return res.content
    elif res.status_code == 404:
        if json.loads(res.content)["errorCode"] == "DISCUSSION_NOT_FOUND":
            print("DISCUSSION_NOT_FOUND")
            return None
        else:
            print(res.content)
    else:
        print(res.content)
        return None


def load_short_ids_from_path(path):
    j = json.load(open(path, "rb"))
    r = j['response']['results']
    id_list = []
    for item in r:
        id = item['id']
        shortUrl = item['fields']['shortUrl']
        id_list.append((id, shortUrl))

    short_ids = []
    for id, shortUrl in id_list:
        short_ids.append(shortUrl[14:].strip())

    return short_ids
