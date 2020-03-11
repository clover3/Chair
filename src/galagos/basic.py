import json
from collections import Counter


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
    q_group = dict()
    for line in open(path, "r"):
        q_id, _, doc_id, rank, score, _ = line.split()
        if q_id not in q_group:
            q_group[q_id] = list()
        q_group[q_id].append((doc_id, int(rank), float(score)))
    return q_group


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


