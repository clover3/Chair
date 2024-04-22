from trainer_v2.per_project.transparency.misc_common import save_tsv
from trainer_v2.per_project.transparency.mmp.bias.new_bias.path_helper import load_product_category_list, \
    get_brand_name_manual_filtered_path, get_rel_table_score_path, selected_brand_score_path
import os



def main():
    q_terms = load_product_category_list()
    for idx_s, term in q_terms:
        print("q_term", term)

        b_name_path = get_brand_name_manual_filtered_path(term)

        if not os.path.exists(b_name_path):
            continue

        score_path = get_rel_table_score_path(term)
        d_terms = dict([t.split() for t in open(score_path, "r")])

        out_table = []
        for line in open(b_name_path, "r"):
            brand_name = line.strip()
            out_table.append([brand_name, d_terms[brand_name]])

        save_tsv(out_table, selected_brand_score_path(term))



if __name__ == "__main__":
    main()
