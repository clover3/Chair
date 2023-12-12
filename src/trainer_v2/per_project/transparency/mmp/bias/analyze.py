from cpath import output_path
from list_lib import right
from misc_lib import path_join, average, get_second
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.mmp.bias.common import contain_any
from trainer_v2.per_project.transparency.mmp.bias.exp1.run_inference_w_keyword_swap import load_car_bias_exp_resource


def read_score_log():
    dir_path = path_join(output_path, "mmp", "bias", "score_log")
    for i in range(1, 20):
        yield from tsv_iter(path_join(dir_path, f"{i}.tsv"))


def analyze_inner(passages, query_list, score_log, term_list, term_list_set):
    seen_doc = set()
    for row in score_log:
        doc_idx = int(row[0])
        if doc_idx in seen_doc:
            continue

        seen_doc.add(doc_idx)
        q_idx = int(row[1])
        if contain_any(query_list[q_idx], term_list_set):
            continue

        scores = list(map(float, row[2:]))
        mean_scores = average(scores)
        doc_tokens = passages[doc_idx].split()

        originally_matched_terms = [(idx, term) for idx, term in enumerate(term_list) if term in doc_tokens]

        if max(scores) < 0:
            continue
        print("Relevant")
        entry_list = []
        for idx, s in enumerate(scores):
            entry = [term_list[idx], scores[idx]]
            entry_list.append(entry)
        entry_list.sort(key=get_second, reverse=True)
        # if high indices has something other than originally matched terms, it is suspicious

        entity_to_score = dict(entry_list)

        entity_to_score["ford"] > entity_to_score["honda"]
        print(f"query {q_idx}: {query_list[q_idx]}")
        print(f"passage {doc_idx}: {passages[doc_idx]}")
        print("Original: " + ", ".join(right(originally_matched_terms)))
        for idx, term in originally_matched_terms:
            print(term, scores[idx])
        print("High scored: ")
        s = ", ".join(["{0} ({1:.2f})".format(term, score) for term, score in entry_list])
        print(s)
        print()


def main():
    score_log = read_score_log()
    passages, query_list, term_list_set, term_list = load_car_bias_exp_resource()

    analyze_inner(passages, query_list, score_log, term_list, term_list_set)



if __name__ == "__main__":
    main()