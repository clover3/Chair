
from cpath import output_path
from misc_lib import path_join
from utils.open_ai_api import GPTPromptHelper


def read_voca():
    lines = open(path_join(output_path, "mmp", "lucene_krovetz", "freq100K.txt"), "r").readlines()
    return [t.strip() for t in lines]




def main():
    voca = read_voca()
    prompt = "{}\n\n List words in the above list which be considered as a product category. print the word as in the list"
    gpt = GPTPromptHelper(prompt)
    start = 1700
    step = 100
    end = 10000
    log_save_path = path_join(output_path, "mmp", "bias", "product_category_like_gpt.txt")
    out_log = open(log_save_path, "a")

    for i in range(start, end, step):
        items = voca[i:i+step]
        core = "\n".join(items)
        ret = gpt.request(core)
        print(ret)

        out_log.write("{}:{}\n".format(i, i+step))
        out_log.write(ret + "\n")



if __name__ == "__main__":
    main()