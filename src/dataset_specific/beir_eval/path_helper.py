from cpath import output_path
from misc_lib import path_join


def get_beir_working_dir(dataset):
    return path_join(output_path, "beir", dataset)


def get_beir_inv_index_path(dataset):
    return path_join(get_beir_working_dir(dataset), "inv_index")


def get_beir_df_path(dataset):
    return path_join(get_beir_working_dir(dataset), "df")


def get_beir_dl_path(dataset):
    return path_join(get_beir_working_dir(dataset), "dl")


def get_json_qres_save_path(run_name):
    save_dir = path_join(output_path, "json_res")
    return path_join(save_dir, run_name + ".json")

