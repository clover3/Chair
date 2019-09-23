import os
import csv
from collections import defaultdict
from summarization.tokenizer import tokenize
from models.classic.stopword import load_stopwords

root_dir = "C:\work\Data\CKB annotation"



def is_sorted(l):
    before = -99
    for elem in l:
        if not before < elem:
            return False
        before = elem
    return True

def load_claim_gen(path):
    f = open(path, "r", encoding="utf-8")
    data = []
    for row in csv.reader(f):
        data.append(row)

    head = list(data[0])

    col_a_id = head.index("AssignmentId")
    col_idx_status = head.index("AssignmentStatus")
    col_url_idx = head.index("Input.url")

    col_statement_idx = []
    for i in range(1,6):
        idx = head.index("Answer.statement{}".format(i))
        col_statement_idx .append(idx)

    url2id = {}
    id2url = {}
    url_id_idx = 0

    parsed_data = []
    for entry in data[1:]:

        if entry[col_idx_status] in ["Rejected", "ManReject"]:
            continue

        d_entry = {}
        d_entry['a_id'] = entry[col_a_id]
        url = entry[col_url_idx]
        if url not in url2id:
            id2url[url_id_idx] = url
            url2id[url] = url_id_idx
            url_id_idx += 1
        statements = []
        for i in range(5):
            idx = col_statement_idx[i]
            statements.append(entry[idx])

        d_entry['statements'] = statements
        d_entry['url'] = url
        d_entry['url_id'] = url2id[url]
        parsed_data.append(d_entry)

    return parsed_data, id2url





def average(iters):
    l = list(iters)
    return sum(l) / len(l)

def merge_batch(parsed_data):
    stopwords = load_stopwords()

    all_annotations = defaultdict(list)
    for entry in parsed_data:
        url_id = entry['url_id']
        all_annotations[url_id].extend(entry['statements'])

    def get_dist(text1, text2):
        tokens1 = tokenize(text1, stopwords)
        tokens2 = tokenize(text2, stopwords)

        common = set(tokens1).intersection(set(tokens2))
        n_common = len(common)
        return (n_common/ len(tokens1)) * (n_common/len(tokens2) )

    result = dict()
    dist_thres = 0.5
    for key in all_annotations:
        annot_list = all_annotations[key]
        n = len(annot_list)
        for idx1 in range(n):
            for idx2 in range(idx1+1, n):
                annot1 = annot_list[idx1]
                annot2 = annot_list[idx2]
                dist = get_dist(annot1, annot2)

                if dist > dist_thres :
                    print("Diff : ")
                    print(annot1)
                    print(annot2)

        result[key] = annot_list
    return result


def generate_merged():
    data_name = "claim generation 1"
    path = os.path.join(root_dir, data_name, "Batch_3744706_batch_results.csv")
    r, id2url = load_claim_gen(path)
    result = merge_batch(r)
    out_name = "merged.res.csv"
    out_path = os.path.join(root_dir, data_name, out_name)
    fout = open(out_path, "w", newline="")
    result_writer = csv.writer(fout)


    for url_id in result:
        for s in result[url_id]:
            url = id2url[url_id]
            result_writer.writerow([url_id, url, s])

    out_path = os.path.join(root_dir, data_name, "claims.csv")
    fout2 = open(out_path, "w", newline="")
    result_writer = csv.writer(fout2)

    for url_id in result:
        for s in result[url_id]:
            result_writer.writerow([s])


if __name__ == "__main__":
    generate_merged()



