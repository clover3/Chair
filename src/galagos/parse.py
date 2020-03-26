import json
import string
from collections import Counter
from typing import Iterator, Dict, List, TypeVar, NamedTuple

from list_lib import flatten, right
from misc_lib import group_by


class GalagoDocRankEntry(NamedTuple):
    doc_id: str
    rank: int
    score: float


class GalagoPassageRankEntry(NamedTuple):
    doc_id : str
    st: int
    ed: int
    rank: int
    score: float


GalagoRankEntry = TypeVar('GalagoRankEntry' , GalagoDocRankEntry, GalagoPassageRankEntry)


def galago_judgement_parse_line(line):
    first_space = line.find(' ')
    second_space = line.find(' ', first_space+1)

    last_space = line.rfind(' ')
    second_last_space = line.rfind(' ', 0, last_space)
    third_last_space = line.rfind(' ', 0, second_last_space)

    q_id = line[:first_space]
    doc_id = line[second_space+1:third_last_space]
    rank = line[third_last_space+1:second_last_space]
    score = line[second_last_space+1:last_space]
    return q_id, doc_id, rank, score


def load_galago_judgement2(path):
# Sample Format : 475287 Q0 LA053190-0016_1274 1 15.07645119 galago
    q_group = dict()
    for line in open(path, "r"):
        q_id, doc_id, rank, score = galago_judgement_parse_line(line)
        if q_id not in q_group:
            q_group[q_id] = list()
        q_group[q_id].append((doc_id, int(rank), float(score)))
    return q_group


def load_galago_ranked_list(path):
    # Sample Format : 475287 Q0 LA053190-0016_1274 1 15.07645119 galago
    line_itr = open(path, "r")
    return parse_galago_ranked_list(line_itr)


def parse_galago_ranked_list(line_itr: Iterator[str]) -> Dict[str, List[GalagoDocRankEntry]]:
    q_group = dict()
    for line in line_itr:
        q_id, _, doc_id, rank, score, _ = line.split()
        if q_id not in q_group:
            q_group[q_id] = list()
        e = str(doc_id), int(rank), float(score)
        q_group[q_id].append(e)
    return q_group


def parse_galago_passage_ranked_list(line_itr: Iterator[str])\
        -> Dict[str, List[GalagoPassageRankEntry]]:
    q_group = dict()
    for line in line_itr:
        q_id, _, doc_id, rank, score, _, st, ed = line.split()
        if q_id not in q_group:
            q_group[q_id] = list()
        q_group[q_id].append(GalagoPassageRankEntry(doc_id=str(doc_id),
                                                    st=int(st),
                                                    ed=int(ed),
                                                    rank=int(rank),
                                                    score=float(score)))
    return q_group


# Input Dict[Query, ranked_list])
def combine_ranked_list(ranked_list_list):
    data = []
    for d in ranked_list_list:
        for query, ranked_list in d.items():
            data.append((query, ranked_list))

    new_d = {}
    key_fn = lambda x: x[0]
    for key, sub_data in group_by(data, key_fn).items():
        ranked_list = right(sub_data)
        new_d[key] = merge_ranked_list_list(ranked_list)
    return new_d


def load_tf(file_path):
    f = open(file_path, "r", encoding="utf-8")
    lines = f.readlines()
    tf_dict = Counter()
    ctf = int(lines[0])
    for line in lines[1:]:
        tokens = line.split()
        if len(tokens) != 3:
            continue

        word, tf, df = line.split()
        tf = int(tf)
        df = int(df)
        tf_dict[word] = tf

    return ctf, tf_dict


def load_df(file_path):
    f = open(file_path, "r", encoding="utf-8")
    lines = f.readlines()
    df_dict = Counter()
    for line in lines:
        tokens = line.split()
        if len(tokens) != 3:
            continue

        word, tf, df = line.split()
        df = int(df)
        df_dict[word] = df

    return df_dict


def write_query_json(queries, out_path):
    j_queries = []
    for q_id, query in queries:
        j_queries.append({"number": str(q_id), "text": "#combine({})".format(" ".join(query))})

    data = {"queries": j_queries}
    fout = open(out_path, "w")
    fout.write(json.dumps(data, indent=True))
    fout.close()


def parse_doc_jsonl_line(line):
    j = json.loads(line, strict=False)
    html = j['content']
    doc_id = j['id']
    return doc_id, html


def load_queries(query_path):
    j = json.load(open(query_path, "r"))
    queries = j["queries"]
    return queries


def count_query(query_path):
    return len(load_queries(query_path))


def merge_ranked_list_list(ranked_list_lst):
    ranked_list = flatten(ranked_list_lst)
    assert len(ranked_list[0]) == 3
    ranked_list.sort(key=lambda x: x[2], reverse=True)
    return ranked_list


def clean_query(query):
    q_term = []
    spe_chars = set([t for t in string.printable if not t.isalnum()])
    for t in query:
        if t in spe_chars:
            continue
        else:
            q_term.append(t)
    return q_term


def get_query_entry(q_id, query):
    return {"number": str(q_id), "text": "#combine({})".format(" ".join(query))}


def get_query_entry_bm25_anseri(q_id, query):
    return {"number": str(q_id), "text": "#combine(bm25:K=0.9:b=0.4({}))".format(" ".join(query))}


def save_queries_to_file(queries, out_path):
    data = {"queries": queries}
    fout = open(out_path, "w")
    fout.write(json.dumps(data, indent=True))
    fout.close()