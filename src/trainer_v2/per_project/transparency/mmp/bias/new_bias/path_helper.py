from cpath import output_path
from misc_lib import path_join


def get_brand_name_wiki_filtered_path(term):
    brand_names_save = path_join(output_path, "mmp", "bias", "brand",
                                 "brand_names", "wiki_filtered", f"{term}.txt")
    return brand_names_save

def get_brand_name_manual_filtered_path(term):
    brand_names_save = path_join(output_path, "mmp", "bias", "brand",
                                 "brand_names", "manual_filtered", f"{term}.txt")
    return brand_names_save


def selected_brand_score_path(term):
    return path_join(output_path, "mmp", "bias", "brand",
                                 "brand_names", "sel_brand_scores", f"{term}.txt")


def get_rel_table_score_path(term):
    score_path = path_join(output_path, "mmp", "bias", "brand",
                           "product_category_term_scores", f"{term}.txt")
    return score_path


def get_wiki_txt_path(term):
    wiki_txt_path = path_join(output_path, "mmp", "bias", "brand",
                              "brand_names", "wiki", f"{term}.txt")
    return wiki_txt_path


def load_product_category_list():
    log_save_path = get_product_category_list_path()
    lines = open(log_save_path, "r").readlines()
    return [t.strip().split() for t in lines]


def get_product_category_list_path():
    return path_join(output_path, "mmp", "bias", "brand", "product_category_selected.txt")
