from trainer_v2.per_project.transparency.misc_common import read_lines
from cpath import output_path
from misc_lib import path_join
from utils.open_ai_api import ENGINE_GPT_3_5, OpenAIProxy, parse_chat_gpt_response


def main():
    text_path = path_join(output_path, "mmp", "bias", "car_exp", "generic_mention_text.txt")
    lines = read_lines(text_path)
    postfix_question = "\n-\nYou you think this text contains any information that is more relevant to a specific car maker (brand)? Answer Yes/No"
    proxy = OpenAIProxy(ENGINE_GPT_3_5)
    save_path = path_join(output_path, "mmp", "bias", "car_exp", "generic_mention_text_gpt_label.txt")
    f = open(save_path, "w")

    for line in lines:
        prompt = line + postfix_question
        print(prompt)
        res = proxy.request(prompt)
        res_text = parse_chat_gpt_response(res)
        print(res_text)
        f.write(res_text + "\n")


if __name__ == "__main__":
    main()