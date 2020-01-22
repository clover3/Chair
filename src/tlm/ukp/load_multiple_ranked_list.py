import os

from adhoc.galago import load_galago_ranked_list
from data_generator.argmining import ukp
from misc_lib import get_dir_files, group_by, flatten, right


def ukp_ranked_list_name_to_group_key(file_name):
    for topic in ukp.all_topics:
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

    def merge(ranked_list_lst):
        ranked_list = flatten(ranked_list_lst)
        assert len(ranked_list[0]) == 3
        ranked_list.sort(key=lambda x: x[2], reverse=True)
        return ranked_list

    new_d = {}
    key_fn = lambda x: get_key_from_name(x[0])
    for key, sub_data in group_by(data, key_fn).items():
        ranked_list = right(sub_data)
        new_d[key] = merge(ranked_list)

    return new_d


def sydney_get_ukp_ranked_list():
    path = "/home/youngwookim/work/ukp/relevant_docs/clueweb12"
    return load_multiple_ranked_list(path, ukp_ranked_list_name_to_group_key)


