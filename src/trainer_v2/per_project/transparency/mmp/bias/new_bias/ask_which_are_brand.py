
from cpath import output_path
from misc_lib import path_join, exist_or_mkdir
from trainer_v2.per_project.transparency.mmp.bias.new_bias.path_helper import load_product_category_list, \
    get_rel_table_score_path
from utils.open_ai_api import GPTPromptHelper


def main():
    q_terms = load_product_category_list()
    for idx_s, term in q_terms:
        print("q_term", term)
        score_path = get_rel_table_score_path(term)
        exist_or_mkdir(path_join(output_path, "mmp", "bias", "brand",
                               "brand_names"))
        brand_names_save = path_join(output_path, "mmp", "bias", "brand",
                               "brand_names", f"{term}.txt")

        prompt = "{}" + f"\n\n Which of the above are the companies or brand name who makes {term}?" \
                        " print the word as in the list"
        gpt = GPTPromptHelper(prompt)
        d_terms = [t.split()[0] for t in open(score_path, "r")]
        print(d_terms)
        start = 0
        step = 100
        end = min(len(d_terms), 1000)
        out_log = open(brand_names_save, "w")

        print("{} items".format(len(d_terms)))
        for i in range(start, end, step):
            items = d_terms[i:i+step]
            core = "\n".join(items)
            ret = gpt.request(core)
            print(ret)
            out_log.write("{}:{}\n".format(i, i+step))
            out_log.write(ret + "\n")
            break


if __name__ == "__main__":
    main()