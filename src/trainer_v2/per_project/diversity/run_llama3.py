from cpath import output_path
from misc_lib import path_join
from trainer_v2.per_project.diversity.prompt_maker import get_prompt2, run_request
from utils.open_ai_api import DeepInfraOpenAIProxy


def main():
    factory = get_prompt2()
    response_path = path_join(output_path, "diversity", "model_responses.jsonl")

    eval_save_path = path_join(output_path, "diversity", "eval_run_llama3-70B", "1.jsonl")
    model = "meta-llama/Meta-Llama-3-70B-Instruct"
    proxy = DeepInfraOpenAIProxy(model)
    proxy.limit_per_msg = 500 * 4 * 10
    run_request(eval_save_path, factory, proxy, response_path)




if __name__ == "__main__":
    main()