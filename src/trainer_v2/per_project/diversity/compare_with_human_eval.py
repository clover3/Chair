import json
from collections import defaultdict

from numpy import isnan
from scipy.stats import pearsonr
from cpath import output_path
from iter_util import load_jsonl
from trainer_v2.per_project.diversity.response_parser import parse_eval_res_from_jsonl, parse_one_line, \
    parse_first_success_line
from misc_lib import path_join, average, get_dir_files
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


def load_human_eval2() -> dict[ModelName, dict[int, float]]:
    eval_save_dir = path_join(output_path, "diversity", "human_eval")
    parsed_j = {}
# {query_id": 4, "LM": "openchat_3.5", "subtopic_covered_human": 0, "subtopic_total": 6}
    for file_path in get_dir_files(eval_save_dir):
        j = load_jsonl(file_path)
        model_name = ""
        score_d = {}
        for entry in j:
            query_id = entry["query_id"]
            model_name = entry["LM"]
            abs_score = entry["subtopic_covered_human"]
            total_value = entry["subtopic_total"]
            score_d[query_id] = abs_score / total_value
        parsed_j[model_name] = score_d
    return parsed_j

def per_topic_corr(
        human_eval: dict[str, dict[int, float]],
        llm_eval, model_names):
    r_value_list = []
    for llm_eval_e in llm_eval:
        qid = llm_eval_e["qid"]
        per_topic = []
        for model_name in model_names:
            h_score: float = human_eval[model_name][qid]
            per_topic.append((h_score, model_name))

        model_to_score = {}
        for score, model_name in per_topic:
            model_to_score[model_name] = score

        score_l = []
        score_h = []
        for model_name in model_names:
            score_l.append(llm_eval_e["score"][model_name])
            score_h.append(model_to_score[model_name])

        r_value, p_value = pearsonr(score_l, score_h)
        if not isnan(r_value):
            r_value_list.append(r_value)

    print("Per topic model score corr\t{}".format(average(r_value_list)))


def print_pearson(scores1, scores2):
    r, p = pearsonr(scores1, scores2)
    print(f"{r}\t{p}")


def flat_corr(human_eval, llm_eval, model_names):
    flat_h_score = []
    flat_l_score = []
    for llm_eval_e in llm_eval:
        qid = llm_eval_e["qid"]
        for model_name in model_names:
            h_score: float = human_eval[model_name][qid]
            l_score: float = llm_eval_e["score"][model_name]
            flat_h_score.append(h_score)
            flat_l_score.append(l_score)

    print("Flat Correlation")
    print_pearson(flat_l_score, flat_h_score)


def model_corr(human_eval, llm_eval, model_names):

    per_model_llm_eval_scores = defaultdict(list)
    per_model_human_eval_scores = defaultdict(list)

    for llm_eval_e in llm_eval:
        qid = llm_eval_e["qid"]
        for model_name in model_names:
            per_model_llm_eval_scores[model_name].append(llm_eval_e["score"][model_name])

        for model_name in model_names:
            per_model_human_eval_scores[model_name].append(human_eval[model_name][qid])

    l_score = [average(per_model_llm_eval_scores[model_name]) for model_name in model_names]
    h_score = [average(per_model_human_eval_scores[model_name]) for model_name in model_names]

    print("Per model score")
    table = []
    head = ["eval_method"] + model_names
    table.append(head)
    table.append(["Human eval"] + h_score)
    table.append(["LLM eval / 2"] + [t / 2 for t in l_score])
    table.append(["LLM eval"] + l_score)
    print_table(table)
    print_pearson(l_score, h_score)



def main():
    human_eval: dict[ModelName, dict[int, float]] = load_human_eval2()
    eval_save_path = path_join(output_path, "diversity", "eval_run3", "concat.jsonl")
    llm_eval = parse_eval_res_from_jsonl(eval_save_path, parse_one_line)
    print(len(llm_eval), "topics are evalauted")
    model_names = list(llm_eval[0]["score"].keys())
    print(model_names)
    per_topic_corr(human_eval, llm_eval, model_names)
    flat_corr(human_eval, llm_eval, model_names)
    model_corr(human_eval, llm_eval, model_names)


def main2():
    human_eval: dict[ModelName, dict[int, float]] = load_human_eval2()
    eval_save_path = path_join(output_path, "diversity", "eval_run_llama3-70B", "1_fix.jsonl")
    def parse_text(arg1, arg2):
        return parse_first_success_line(4, arg1, arg2)

    llm_eval = parse_eval_res_from_jsonl(eval_save_path, parse_text)
    print(len(llm_eval), "topics are evalauted")
    model_names = list(llm_eval[0]["score"].keys())
    print(model_names)
    per_topic_corr(human_eval, llm_eval, model_names)
    flat_corr(human_eval, llm_eval, model_names)
    model_corr(human_eval, llm_eval, model_names)


if __name__ == "__main__":
    main()
    main2()