from misc_lib import path_join


def get_mmp_train_grouped_sorted_path(job_no):
    quad_tsv_path = path_join("data", "msmarco", "passage", "group_sorted_10K", str(job_no))
    return quad_tsv_path


def get_mmp_grouped_sorted_path(split, job_no):
    if split == "train":
        return get_mmp_train_grouped_sorted_path(job_no)
    else:
        quad_tsv_path = path_join("data", "msmarco", "passage", f"{split}_group_sorted_10K", str(job_no))
    return quad_tsv_path
