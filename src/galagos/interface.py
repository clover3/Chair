# This file contains functions to actually execute galago binary and deliver the results
# The functions to parse the results are in galagos.parse

import os
import subprocess
import time
from collections import Counter
from subprocess import PIPE
from typing import List, Dict

from cpath import output_path
from galagos.parse import save_queries_to_file, parse_galago_ranked_list, parse_galago_passage_ranked_list
from galagos.types import SimpleRankedListEntry, GalagoPassageRankEntry
from misc_lib import exist_or_mkdir
from taskman_client.sync import JsonTiedDict

dyn_query_dir = os.path.join(output_path, "dyn_query")
exist_or_mkdir(dyn_query_dir)
info_path = os.path.join(dyn_query_dir, "info.json")
task_info = JsonTiedDict(info_path)


class DocQuery(Dict):
    pass


class PassageQuery(Dict):
    pass


def get_new_query_json_path() -> str:
    last_query_file_idx = get_last_query_file_idx()
    new_query_id = last_query_file_idx + 1
    task_info.last_task_id = new_query_id
    return get_json_path_for_idx(new_query_id)


def get_last_query_file_idx():
    init_id = max(task_info.last_id(), 0)
    id_idx = init_id
    while os.path.exists(get_json_path_for_idx(id_idx)):
        id_idx += 1

    return id_idx - 1


def get_json_path_for_idx(idx) -> str:
    return os.path.join(dyn_query_dir, "{}.json".format(idx))


def get_doc(index_path: str, doc_id: str) -> str:
    # issue galago command
    cmd = ["galago",
           "doc",
           "--index=" + index_path,
           '--id=' + doc_id
           ]
    p = subprocess.Popen(cmd,
                         stdout=PIPE,
                         stderr=PIPE,
                         )
    # wait , read pipe
    file_content = p.communicate()
    return file_content[0].decode()


def get_doc_jsonl(index_path, doc_id) -> List[str]:
    # issue galago command
    cmd = ["galago",
           "get-docs-jsonl",
           "--index=" + index_path,
           '--eidList=' + doc_id
           ]
    p = subprocess.Popen(cmd,
                         stdout=PIPE,
                         stderr=PIPE,
                         )
    # wait , read pipe
    file_content = p.communicate()
    return file_content[0].decode().splitlines()


# queries : List[ "num":query_id, "text": query_str ]
# output : Dict[Query -> List(doc_id, score, rank)]
def send_doc_queries(index_path, num_result, queries, timeout=3600) -> Dict[str, List[SimpleRankedListEntry]]:
    lines = send_queries_inner(index_path, num_result, queries, timeout)
    return parse_galago_ranked_list(lines)


def send_queries_passage(index_path: str,
                         num_result: int,
                         queries: List[PassageQuery],
                         timeout: int = 3600,
                         ) \
        -> Dict[str, List[GalagoPassageRankEntry]]:
    lines = send_queries_inner(index_path, num_result, queries, timeout)
    return parse_galago_passage_ranked_list(lines)


def send_queries_inner(index_path, num_result, queries, timeout) -> List[str]:
    query_path = get_new_query_json_path()
    # save query to file
    save_queries_to_file(queries, query_path)
    # issue galago command
    cmd = ["galago",
           "threaded-batch-search",
           "--requested=" + str(num_result),
           "--index=" + index_path, query_path]
    os.environ['PYTHONUNBUFFERED'] = "1"
    temp_outpath = query_path + ".output"
    out_file = open(temp_outpath, "w")
    proc = subprocess.Popen(cmd,
                            stdout=out_file,
                            stderr=PIPE,
                            universal_newlines=True,
                            )
    # wait , read pipe
    prev_num_remain = 999999
    last_update_time = time.time()
    try:
        while proc.poll() is None:
            line = proc.stderr.readline()
            if line.startswith("INFO: Still running..."):
                st = len("INFO: Still running...")
                tokens = line[st:].split()
                num_remain = int(tokens[0])
                if num_remain != prev_num_remain:
                    print(line, end='')
                    prev_num_remain = num_remain
                    last_update_time = time.time()

            if time.time() - last_update_time > timeout:
                break
    except subprocess.TimeoutExpired:
        proc.kill()
    out_file.close()

    file_content = open(temp_outpath, "r").read()
    lines: List[str] = file_content.splitlines()
    return lines


def format_query_bm25(query_id: str,
                      tokens: List[str], k=0) -> DocQuery:

    tokens = [t.replace(".", "") for t in tokens]
    q_str_inner = " ".join(["#bm25:K={}({})".format(k, t) for t in tokens])
    query_str = "#combine({})".format(q_str_inner)
    return DocQuery({
        'number': query_id,
        'text': query_str
    })


def format_query_simple(query_id: str, tokens: List[str]) -> DocQuery:
    tokens = [t.replace(".", "") for t in tokens]
    query_str = " ".join(tokens)
    return DocQuery({
        'number': query_id,
        'text': query_str
    })



def format_passage_query(query_id: str,
                         tokens: List[str],
                         k=0) -> PassageQuery:
    q_str_inner = " ".join(["#bm25:K={}({})".format(k, t) for t in tokens])
    query_str = "#combine({})".format(q_str_inner)
    return PassageQuery({
        'number': query_id,
        'text': query_str,
        "passageQuery": True,
        "passageSize": 200,
        "passageShift": 100,
    })


def write_queries_to_files(n_query_per_file, out_dir, queries: List[DocQuery]):
    i = 0
    while i * n_query_per_file < len(queries):
        st = i * n_query_per_file
        ed = (i + 1) * n_query_per_file
        out_path = os.path.join(out_dir, "{}.json".format(i))
        save_queries_to_file(queries[st:ed], out_path)
        i += 1


def counter_to_galago_query(query_id: str, q_tf: Counter, k=0) -> DocQuery:
    def get_weight(q_term):
        f = q_tf[q_term]
        return "{0:.2f}".format(f)

    keys = q_tf.keys()
    combine_weight = ":".join(["{}={}".format(idx, get_weight(t)) for idx, t in enumerate(keys)])

    q_str_inner = " ".join(["#bm25:K={}({})".format(k, t) for t in keys])
    query_str = "#combine:{}({})".format(combine_weight, q_str_inner)
    return DocQuery({
        'number': query_id,
        'text': query_str
    })