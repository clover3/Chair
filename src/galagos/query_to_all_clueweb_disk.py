import os

from cpath import src_path, project_root
from galagos.basic import load_galago_ranked_list, load_queries
from misc_lib import exist_or_mkdir, lmap
from sydney_clueweb import clue_path

threaded_search_sh_path = os.path.join(src_path, "sh_script", "threaded_search.sh")


def insert_argument(content, arg_map):
    for key, arg_val in arg_map.items():
        content = content.replace(key, arg_val)

    return content


def get_query_file_base_name(full_path):
    query_base_name = os.path.basename(full_path)

    if query_base_name.endswith(".json"):
        query_base_name = query_base_name[:-5]
    return query_base_name


def send(list_query_files, disk_list, job_name, out_root):
    base_file = open(threaded_search_sh_path, "r").read()

    for query_path in list_query_files:
        for disk_name in disk_list:
            query_base_name = get_query_file_base_name(query_path)
            index_path = os.path.join(clue_path.index_dir, disk_name)
            file_name = "{}_{}".format(disk_name, query_base_name)
            out_path = os.path.join(out_root, file_name + ".txt")
            arg = {
                '${index_path}':index_path,
                '${query_file}':query_path,
                '${outpath}': out_path,
            }
            new_content = insert_argument(base_file, arg)
            sh_dir_path = os.path.join(project_root, "script", job_name)
            exist_or_mkdir(sh_dir_path)
            sh_path = os.path.join(sh_dir_path, file_name + ".sh")
            open(sh_path, "w").write(new_content)
            sh_cmd = "sbatch " + sh_path
            os.system(sh_cmd)


def verify_ranked_list(out_path, queries):
    n_query = len(queries)
    file_name = os.path.basename(out_path)
    ranked_list_d = load_galago_ranked_list(out_path)
    if len(ranked_list_d) < n_query:
        print("{} has only {} queries, expected {}".format(file_name, len(ranked_list_d), n_query))
        found_query_ids = set(ranked_list_d.keys())
        queries_d = dict(lmap(lambda x:(x["number"], x["text"]), queries))
        expected_query_ids = lmap(lambda x:x["number"], queries)
        not_found_query_ids = list([t for t in expected_query_ids if t not in found_query_ids])
        for query_id in not_found_query_ids:
            print("Not found: ", queries_d[query_id])





def verify_result(list_query_files, disk_list, out_root):
    for query_path in list_query_files:
        for disk_name in disk_list:
            try:
                query_base_name = get_query_file_base_name(query_path)
                file_name = "{}_{}".format(disk_name, query_base_name)
                out_path = os.path.join(out_root, file_name + ".txt")
                verify_ranked_list(out_path, load_queries(query_path))
            except FileNotFoundError as e:
                print(e)

