import json
from collections import defaultdict

from numpy import isnan
from scipy.stats import pearsonr
from cpath import output_path
from dataset_specific.clueweb.response_parser import parse_eval_res
from misc_lib import path_join, average
from tab_print import print_table

ModelName = str
def load_human_eval() -> dict[ModelName, list[float]]:
    eval_save_path = path_join(output_path, "diversity", "human_eval.json")
    j = json.load(open(eval_save_path, "r"))

    parsed_j = {}
    for entry in j:
        abs_score = entry["humanValues"]
        total_value = entry["totalValues"]
        assert len(abs_score) == len(total_value)
        parsed_j[entry["model"]] = [a/b for a, b in zip(abs_score, total_value)]
    return parsed_j


def main():
    human_eval: dict[ModelName, list[float]] = load_human_eval()
    eval_save_path = path_join(output_path, "diversity", "eval_run2", "concat.jsonl")
    llm_eval = parse_eval_res(eval_save_path)
    model_names = list(llm_eval[0]["rank"].keys())
    print(model_names)
    per_topic_corr(human_eval, llm_eval, model_names)
    flat_corr(human_eval, llm_eval, model_names)
    model_corr(human_eval, llm_eval, model_names)


def per_topic_corr(human_eval, llm_eval, model_names):
    r_value_list = []
    for idx, llm_eval_e in enumerate(llm_eval):
        per_topic = []
        for model_name in model_names:
            h_score: float = human_eval[model_name][idx]
            per_topic.append((h_score, model_name))
        per_topic.sort(key=lambda x: x[0], reverse=True)

        model_to_rank = {}
        for rank, (score, model_name) in enumerate(per_topic):
            model_to_rank[model_name] = rank


        rank_l = []
        rank_h = []
        for model_name in model_names:
            rank_l.append(llm_eval_e["rank"][model_name])
            rank_h.append(model_to_rank[model_name])

        r_value, p_value = pearsonr(rank_l, rank_h)
        r_value_list.append(r_value)

    print("Per topic model rank corr")
    print(average(r_value_list))



def flat_corr(human_eval, llm_eval, model_names):
    flat_h_score = []
    flat_l_score = []
    for idx, llm_eval_e in enumerate(llm_eval):
        for model_name in model_names:
            h_score: float = human_eval[model_name][idx]
            l_score: float = llm_eval_e["score"][model_name]
            flat_h_score.append(h_score)
            flat_l_score.append(l_score)

    print(pearsonr(flat_l_score, flat_h_score))


def model_corr(human_eval, llm_eval, model_names):
    h_score = [average(human_eval[model_name]) for model_name in model_names]

    per_model_scores = defaultdict(list)
    for idx, llm_eval_e in enumerate(llm_eval):
        for model_name in model_names:
            per_model_scores[model_name].append(llm_eval_e["score"][model_name])

    l_score = [average(per_model_scores[model_name]) for model_name in model_names]
    print("Per model score")
    table = []
    head = ["eval_method"] + model_names
    table.append(head)
    table.append(["Human eval"] + h_score)
    table.append(["LLM eval / 2"] + [t / 2 for t in l_score])
    table.append(["LLM eval"] + l_score)
    print_table(table)
    print(pearsonr(l_score, h_score))


if __name__ == "__main__":
    main()