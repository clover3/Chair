import pickle

from misc_lib import get_dir_files
from tlm.ukp.load_multiple_ranked_list import load_multiple_ranked_list, ukp_ranked_list_name_to_group_key, \
    nc_ranked_list_name_to_group_key


def ukp_load_tokens_for_topic(topic):
    token_path = "/mnt/nfs/work3/youngwookim/data/stance/clueweb12_10000_tokens/"
    return load_tokens_for_topic(token_path, topic)


def ukp_load_tokens_for_topic_from_shm(topic):
    token_path = "/dev/shm/"
    return load_tokens_for_topic(token_path, topic)


def non_cont_load_tokens_for_topic(topic):
    token_path = "/mnt/nfs/work3/youngwookim/data/stance/clueweb12_10000_new_tokens/"
    return load_tokens_for_topic(token_path, topic)


def load_tokens_for_topic(token_path, topic):
    d = {}
    for path in get_dir_files(token_path):
        if topic.replace(" ", "_") in path:
            data = pickle.load(open(path, "rb"))
            if len(data) < 10000:
                print("{} has {} data".format(path, len(data)))
            d.update(data)
    print("Loaded {} docs for {}".format(len(d), topic))
    return d


def dev_pretend_ukp_load_tokens_for_topic(topic):
    token_path = "/mnt/nfs/work3/youngwookim/data/stance/clueweb12_10000_tokens/"
    return dev_pretend_load_tokens_for_topic(token_path, topic)


def dev_pretend_load_tokens_for_topic(token_path, topic):
    d = {}
    for path in get_dir_files(token_path):
        if topic.replace(" ", "_") in path:
            data = pickle.load(open(path, "rb"))
            if len(data) < 10000:
                print("{} has {} data".format(path, len(data)))
            d.update(data)
        if len(d)> 100:
            break
    print("Loaded {} docs for {}".format(len(d), topic))
    return d


def sydney_get_ukp_ranked_list():
    path = "/home/youngwookim/work/ukp/relevant_docs/clueweb12"
    return load_multiple_ranked_list(path, ukp_ranked_list_name_to_group_key)


def sydney_get_nc_ranked_list():
    path = "/home/youngwookim/work/ukp/relevant_docs/clueweb12_nc"
    return load_multiple_ranked_list(path, nc_ranked_list_name_to_group_key)