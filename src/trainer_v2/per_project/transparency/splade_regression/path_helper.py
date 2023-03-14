from misc_lib import path_join


def partitioned_triplet_path_format_str():
    text_path_format_str = path_join("data", "msmarco", "splade_triplets_partitioned", "triplet{0:06d}")
    return text_path_format_str

