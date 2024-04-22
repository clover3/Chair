import os
from cpath import output_path
from misc_lib import path_join
import re
from scratch.chatgpt_stuff.segment_text import merge_segments_to_limit
from utils.open_ai_api import OpenAIProxy, ENGINE_GPT_3_5, ENGINE_GPT4


def enumerate_files(path):
    # List to hold file names
    files = []

    # Check if the path exists and is a directory
    if os.path.exists(path) and os.path.isdir(path):
        # List all entries in the directory
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)
            # Check if the entry is a file and add it to the list
            if os.path.isfile(full_path):
                files.append(full_path)
    else:
        print(f"The path {path} is not a valid directory.")

    return files


def drop_comment(s):
    lines = s.split("\n")
    return "\n".join([l for l in lines if not l.startswith(r"%")])


class GPTChecker:
    def __init__(self):
        self.open_ai_proxy = OpenAIProxy(ENGINE_GPT4)
        self.prompt_prefix = "List any spelling mistakes or grammar mistakes in the following text. If no error, say NoError "

    def check(self, text):
        prompt = self.prompt_prefix + ":\n" + text
        ret = self.open_ai_proxy.request_get_text(prompt)

        cleaned = re.sub("[ .]", '', ret)
        if "NoError".lower() in cleaned.lower():
            return None
        else:
            return ret


def main():
    dir_path= r"C:\Users\leste\Downloads\Dissertation"

    start_after = (
        r"C:\Users\leste\Downloads\Dissertation\6_0_alignment_main.tex",
        0
    )
    wait_start = start_after is not None
    import datetime
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d_%H_%M_%S.%f')
    log_file_name = os.path.basename(dir_path) + "_" + formatted_time[:-3] + ".txt"
    print(log_file_name)
    log_save_path = path_join(output_path, "gpt_grammar_check", log_file_name)
    f = open(log_save_path, "w")

    checker = GPTChecker()
    max_char = 2000
    file_list = enumerate_files(dir_path)
    file_list.sort()
    for file_path in file_list:
        file_name = os.path.basename(file_path)
        if file_path.endswith(".tex") and file_name[0].isdigit() and file_name[0] != "0":
            print(file_path)
            s = open(file_path, encoding="utf-8").read()
            segments = merge_segments_to_limit(s, max_char)
            segments = map(drop_comment, segments)
            for i, seg in enumerate(segments):
                if (file_path, i) == start_after:
                    print("Now start checking")
                    wait_start = False
                    continue
                if wait_start:
                    continue
                print(f"Segment {i}")
                ret = checker.check(seg)
                if ret is not None:
                    f.write(file_path + "\n")
                    f.write(f"Segment {i}\n")
                    f.write("Segment head: {}\n".format(seg[:30]))
                    f.write(ret + "\n")

                print(ret)


if __name__ == "__main__":
    main()