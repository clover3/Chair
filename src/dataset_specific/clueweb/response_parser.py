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
    eval_save_path = path_join(output_path, "diversity", "eval_run2", "concat.jsonl")
    eval_res = parse_eval_res(eval_save_path)
    print(eval_res)
    parse_eval_res_for_bias_check(eval_save_path)



if __name__ == "__main__":
    main()