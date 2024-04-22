import os.path

from cpath import output_path
from misc_lib import path_join, exist_or_mkdir
from trainer_v2.per_project.transparency.mmp.bias.new_bias.path_helper import get_brand_name_wiki_filtered_path, \
    get_rel_table_score_path, get_wiki_txt_path, load_product_category_list


def filter_brand_name(wiki_txt, terms) -> list[str]:
    ref = wiki_txt.lower().split()

    output = []
    for t in terms:
        if t in ref:
            output.append(t)
    return output


def main():
    q_terms = load_product_category_list()
    for idx_s, term in q_terms:
        print("q_term", term)
        wiki_txt_path = get_wiki_txt_path(term)
        if not os.path.exists(wiki_txt_path):
            continue

        score_path = get_rel_table_score_path(term)
        d_terms = [t.split()[0] for t in open(score_path, "r")]

        exist_or_mkdir(path_join(output_path, "mmp", "bias", "brand",
                               "brand_names"))
        brand_names_save = get_brand_name_wiki_filtered_path(term)

        wiki_txt = open(wiki_txt_path, "r", encoding="utf-8").read()
        filtered_term = filter_brand_name(wiki_txt, d_terms)

        with open(brand_names_save, "w") as f:
            f.write("\n".join(filtered_term))


if __name__ == "__main__":
    main()