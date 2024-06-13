from collections import Counter, defaultdict
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator, Any
from cpath import output_path
from iter_util import load_jsonl
from misc_lib import path_join



def parse_eval_res(eval_save_path) -> list[dict[str, Any]]:
    n_model = 4
    res_list = load_jsonl(eval_save_path)
    parsed_output: list[dict[str, Any]] = []
    rank_list = []
    for r in res_list:
        text = r["eval_llm_response"]

        state = "rank"
        parsed: dict[str, Any] = {"qid": r["qid"]}
        rank_to_name = {}
        for line in text.split("\n"):
            tokens = line.split()
            if state == "rank":
                try:
                    rank_d = {}
                    for rank, res_idx_s in enumerate(tokens):
                        model_name = r["names"][int(res_idx_s)]
                        rank_d[model_name] = rank
                        rank_to_name[rank] = model_name
                except IndexError:
                    if max(map(int, tokens)) == n_model:
                        # Maybe each res_idx is incremented by 1 (1-index instead of 0-index)
                        rank_d = {}
                        for rank, res_idx_s in enumerate(tokens):
                            model_name = r["names"][int(res_idx_s)-1]
                            rank_d[model_name] = rank
                            rank_to_name[rank] = model_name
                    else:
                        raise
                parsed[state] = rank_d
                state = "score"
            elif state == "score":
                parsed[state] = [float(s) for s in tokens]
                score_d = {}
                for rank, score_s in enumerate(tokens):
                    model_name = rank_to_name[rank]
                    score_d[model_name] = float(score_s)
                parsed[state] = score_d
                state = "done"
            else:
                print("Not expected", line)
        parsed_output.append(parsed)
        rank_list.append(parsed["rank"])
    return parsed_output


def parse_eval_res_from_jsonl(eval_save_path, response_parse) -> list[dict[str, Any]]:
    res_list = load_jsonl(eval_save_path)
    parsed_output: list[dict[str, Any]] = []
    for r in res_list:
        parsed: dict[str, Any] = {"qid": r["qid"]}
        text = r["eval_llm_response"]
        score_d = response_parse(r, text)
        parsed["score"] = score_d
        parsed_output.append(parsed)
    return parsed_output


def parse_one_line(r, text):
    score_d = {}
    for token in text.split():
        idx, score_s = token.split(":")
        model_name = r["names"][int(idx) - 1]
        score_d[model_name] = float(score_s)
    return score_d


def parse_first_success_line(n_model, r, text):
    score_d = {}
    for line in text.split("\n"):
        line = line.replace(": ", ":")
        try:
            for token in line.split():
                idx, score_s = token.split(":")
                model_name = r["names"][int(idx) - 1]
                score_d[model_name] = float(score_s)

            if len(score_d) == n_model:
                break
        except ValueError:
            pass
    if len(score_d) != n_model:
        print(text)
        print(score_d)
        print("Expected {} but got {}".format(len(score_d), n_model))

    return score_d


def parse_eval_res_for_bias_check(eval_save_path):
    n_model = 4
    res_list = load_jsonl(eval_save_path)
    count = defaultdict(Counter)
    for r in res_list:
        text = r["eval_llm_response"]
        state = "rank"
        for line in text.split("\n"):
            tokens = line.split()
            if state == "rank":
                for rank, res_idx_s in enumerate(tokens):
                    count[int(res_idx_s)][rank] += 1
                state = "done"

    for res_idx in range(n_model):
        total = sum(count[res_idx].values())
        share = [count[res_idx][rank_idx] / total for rank_idx in range(n_model)]
        print(share)


def main():
    eval_save_path = path_join(output_path, "diversity", "eval_run3", "2.jsonl")
    eval_res = parse_eval_res_from_jsonl(eval_save_path, parse_one_line)
    print(eval_res)


def main2():
    eval_save_path = path_join(output_path, "diversity", "eval_run_llama3-70B", "1_fix.jsonl")

    def parse_text(arg1, arg2):
        return parse_first_success_line(4, arg1, arg2)
    eval_res = parse_eval_res_from_jsonl(eval_save_path, parse_text)
    print(eval_res)



if __name__ == "__main__":
    main2()