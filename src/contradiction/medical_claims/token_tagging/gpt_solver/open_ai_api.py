import os
import openai
from cpath import output_path, data_path
from misc_lib import path_join


def get_open_ai_api_key():
    s = open(path_join(data_path, "openai_api_key.txt"), "r").read().strip()
    return s


class OpenAIProxy:
    def __init__(self, engine):
        openai.organization = "org-H88LVGb8C9Zc6OlcieMZgW6e"
        openai.api_key = get_open_ai_api_key()
        self.engine = engine

    def request(self, prompt):
        obj = openai.api_resources.Completion.create(
            engine=self.engine, prompt=prompt, logprobs=1,
            max_tokens=512
        )
        return obj



def dev():
    engine = "text-davinci-003"
    # engine = "text-ada-001"
    instruction = "Fix the spelling mistakes"
    input_text = "What day of the wek is it?"
    prompt = instruction + "\n\n" + input_text
    obj = openai.api_resources.Completion.create(engine=engine, prompt=prompt, logprobs=1)
    print(obj)
    print(obj['choices'][0])


def parse_log_probs(logprobs):
    tokens = logprobs['tokens']
    token_logprobs = logprobs['token_logprobs']
    text_offset = logprobs['text_offset']
