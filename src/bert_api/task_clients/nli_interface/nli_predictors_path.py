from cpath import at_output_dir


def get_nli_cache_sqlite_path():
    return at_output_dir("nli", "nli_cache.sqlite")


def get_nli14_cache_sqlite_path():
    return at_output_dir("nli", "nli14.sqlite")


def get_pep_cache_sqlite_path():
    return at_output_dir("nli", "pep.sqlite")