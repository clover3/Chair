import os

import data_generator.argmining.ukp_header
from galagos.basic import load_galago_ranked_list, merge_ranked_list_list
from misc_lib import get_dir_files, group_by, right

non_cont_topics = ["hydroponics", "weather", "restaurant", "wildlife_extinction", "james_allan"]

def ukp_ranked_list_name_to_group_key(file_name):
    for topic in data_generator.argmining.ukp_header.all_topics:
        if topic.replace(" ", "_")in file_name:
            return topic

    raise Exception("Not matched" + file_name)

def nc_ranked_list_name_to_group_key(file_name):
    for topic in non_cont_topics:
        if topic.replace(" ", "_")in file_name:
            return topic
    raise Exception("Not matched" + file_name)


def load_multiple_ranked_list(dir_path, get_key_from_name):
    files = get_dir_files(dir_path)

    data = []
    for file_path in files:
        name = os.path.basename(file_path)
        ranked_list_d = load_galago_ranked_list(file_path)
        for query, ranked_list in ranked_list_d.items():
            data.append((name, ranked_list))

    new_d = {}
    key_fn = lambda x: get_key_from_name(x[0])
    for key, sub_data in group_by(data, key_fn).items():
        ranked_list = right(sub_data)
        new_d[key] = merge_ranked_list_list(ranked_list)

    return new_d


