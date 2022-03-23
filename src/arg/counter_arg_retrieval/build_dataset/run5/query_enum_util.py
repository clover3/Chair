from misc_lib import get_dir_files


def get_files_for_each_query(dir_path):
    n_files_expected = 53
    file_list = list(get_dir_files(dir_path))
    if len(file_list) != n_files_expected:
        print("Expected {} but {} files found ".format(n_files_expected, len(file_list)))
    return file_list