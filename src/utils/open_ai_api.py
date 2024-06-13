import os
from collections import Counter
from openai import OpenAI

import openai
from cpath import output_path, data_path
from misc_lib import path_join


ENGINE_INSTRUCT = 'text-davinci-003'
ENGINE_GPT4 = 'gpt-4'
ENGINE_GPT_3_5 = 'gpt-3.5-turbo'


def get_open_ai_api_key():
    s = open(path_join(data_path, "openai_api_key.txt"), "r").read().strip()
    return s


def get_open_ai_api_key_zamani():
    s = open(path_join(data_path, "openai_api_key_zamani.txt"), "r").read().strip()
    return s


def get_deep_infra_api_key():
    s = open(path_join(data_path, "deep_infra_api_key.txt"), "r").read().strip()
    return s


class OpenAIProxy:
    def __init__(self, engine, api_key=None):
        # openai.organization = "org-H88LVGb8C9Zc6OlcieMZgW6e"
        # openai.api_key = get_open_ai_api_key()

        if api_key is None:
            api_key = get_open_ai_api_key()
        self.client = OpenAI(api_key=api_key)

        self.engine = engine
        self.usage = Counter()
        self.limit_per_msg = 5000

    def request(self, prompt):
        if len(prompt) > self.limit_per_msg:
            raise ValueError("{} exceeds limit {} ".format(len(prompt), self.limit_per_msg))
        if self.engine in [ENGINE_INSTRUCT]:
            obj = openai.api_resources.Completion.create(
                engine=self.engine, prompt=prompt, logprobs=1,
                max_tokens=512,
            )
        else:
            messages = [{"role": "user", "content": prompt}]
            obj = self.client.chat.completions.create(
                model=self.engine, messages=messages, timeout=20,
            )
            print(obj)
            n_tokens_used = obj.usage.total_tokens
            self.usage['n_tokens'] += n_tokens_used
            self.usage['n_request'] += 1
        return obj

    def request_get_text(self, prompt):
        obj = self.request(prompt)
        return obj.choices[0].message.content

    def __del__(self):
        print("Usage", self.usage)

class DeepInfraOpenAIProxy(OpenAIProxy):
    def __init__(self, engine):
        # super(DeepInfraOpenAIProxy, self).__init__(engine)
        openai = OpenAI(
            api_key=get_deep_infra_api_key(),
            base_url="https://api.deepinfra.com/v1/openai",
        )
        print(openai.api_key)
        self.client = openai

        self.engine = engine
        self.usage = Counter()
        self.limit_per_msg = 5000


class GPTPromptHelper:
    def __init__(self, prompt_template):
        self.open_ai_proxy = OpenAIProxy(ENGINE_GPT4)
        self.prompt_template = prompt_template

    def request(self, text):
        prompt = self.prompt_template.format(text)
        ret = self.open_ai_proxy.request_get_text(prompt)
        return ret


def dev():
    engine = "text-davinci-003"
    # engine = "text-ada-001"
    instruction = "Fix the spelling mistakes"
    input_text = "What day of the wek is it?"
    prompt = instruction + "\n\n" + input_text
    # obj = openai.api_resources.Completion.create(engine=engine, prompt=prompt, logprobs=1)
    # print(obj)
    # print(obj['choices'][0])


def dev_chat():
    proxy = OpenAIProxy(ENGINE_GPT_3_5)
    res = proxy.request("hi")
    print(res)


def dev_deep_infra_chat():
    model = "meta-llama/Meta-Llama-3-70B-Instruct"
    proxy = DeepInfraOpenAIProxy(model)
    res = proxy.request("Sum the number of days in months that does not have m in its name")
    print(res)



def parse_log_probs(logprobs):
    tokens = logprobs['tokens']
    token_logprobs = logprobs['token_logprobs']
    text_offset = logprobs['text_offset']


def parse_instruct_gpt_response(response: dict) -> str:
    text = response['choices'][0]['text']
    return text


def parse_chat_gpt_response(response: dict) -> str:
    message = response['choices'][0]['message']
    text = message['content']
    return text


def get_parse_gpt_response_fn(model):
    if model in [ENGINE_INSTRUCT]:
        return parse_instruct_gpt_response
    else:
        return parse_chat_gpt_response


def main():
    dev_deep_infra_chat()


if __name__ == "__main__":
    main()

