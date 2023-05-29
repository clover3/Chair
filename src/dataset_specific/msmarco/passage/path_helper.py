from misc_lib import path_join


def get_mmp_grouped_sorted_path(job_no):
    quad_tsv_path = path_join("data", "msmarco", "passage", "group_sorted_10K", str(job_no))
    return quad_tsv_path
