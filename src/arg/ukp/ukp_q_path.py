import os
from typing import List

from base_type import FileName, FilePath
from cpath import output_path, pjoin
from list_lib import lmap

subproject_hub: FilePath = FilePath('/mnt/nfs/work3/youngwookim/data/stance/ukp_para_query')


query_dir_format = os.path.join(output_path, "perspective_{}_claim_perspective_query_k0")

num_query_file: int = 408


def get_query_dir(query_collection_id) -> FilePath:
    out_dir = pjoin(output_path, FileName("ukp_query_{}".format(query_collection_id)))
    return out_dir


def get_query_file(query_collection_id, i) -> FilePath:
    return pjoin(get_query_dir(query_collection_id), FileName("{}.json".format(i)))


def get_query_file_for_split(split, i):
    return get_query_file(query_dir_format.format(split), i)


def get_ranked_list_save_dir(q_config_id):
    return pjoin(subproject_hub, FileName("{}_q_res".format(q_config_id)))


def get_all_query_file_names(query_collection_id) -> List[FilePath]:
    query_files = lmap(lambda i: get_query_file(query_collection_id, i), range(0, num_query_file))
    return query_files

##