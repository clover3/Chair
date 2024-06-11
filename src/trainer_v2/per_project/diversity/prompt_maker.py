import json
import random

from dataset_specific.clueweb.query_loader import TrecQuery, load_queries, all_years
from iter_util import load_jsonl
from cpath import output_path
from misc_lib import path_join, group_by
from utils.open_ai_api import OpenAIProxy, get_open_ai_api_key_zamani, ENGINE_GPT4


class PromptFactory:
    def __init__(self, format_str):
        self.format_str = format_str

    def make(self, query, response_list: list[str]):
        head_prompt = self.format_str.format(query, len(response_list))

        segs = []
        for idx, r in enumerate(response_list):
            part = f"Response {idx}: \n" + r
            segs.append(part)

        return head_prompt + "\n" + "\n".join(segs)


def main():
    prompt_format = "The following are response to the query. Rank them by diversity and score them in 0 to 1 scale." \
    "list the response id (only number) from most diverse to list diverse, separated by space. In the next line list the scores for each of them without id.\n" \
    "---- \n" \
    "query: {} \n"
    factory = PromptFactory(prompt_format)
    response_path = path_join(output_path, "diversity", "model_responses.jsonl")
    eval_save_path = path_join(output_path, "diversity", "eval_run2", "2.jsonl")

    open_ai = OpenAIProxy(ENGINE_GPT4, get_open_ai_api_key_zamani())
    open_ai.limit_per_msg = 500 * 4 * 10
    queries: list[TrecQuery] = load_queries(all_years)
    query_d = {q.query_id: q.desc_query for q in queries}
    responses = load_jsonl(response_path)

    response_grouped = group_by(responses, lambda x: x['query_id'])
    all_qids = list(response_grouped.keys())
    all_qids.sort()
    todo = all_qids[3:20]

    f_out = open(eval_save_path, "w")
    for qid in todo:
        res_list = response_grouped[qid]
        q_text = query_d[str(qid)]
        random.shuffle(res_list)
        names = [r["LM"] for r in res_list]
        res_text_list = [r["model_response"] for r in res_list]

        # print("n words: ", [len(text.split()) for text in res_text_list])
        prompt = factory.make(q_text, res_text_list)
        print(names)
        print("Qid", qid)
        print(prompt)
        print("\n")

        eval_response = open_ai.request_get_text(prompt)
        log = {
            "qid": qid,
            "names": names,
            "eval_llm_response": eval_response
        }
        print(eval_response)
        f_out.write(json.dumps(log) + "\n")




if __name__ == "__main__":
    main()


