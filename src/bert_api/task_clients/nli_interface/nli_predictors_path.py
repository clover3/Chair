from cpath import at_output_dir


def get_nli_cache_sqlite_path():
    return at_output_dir("nli", "nli_cache.sqlite")