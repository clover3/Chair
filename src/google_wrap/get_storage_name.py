import os


def get_storage_name():
    if "storage_name" in os.environ:
        storage_name = os.environ["storage_name"]
    else:
        storage_name = "clovertpu"
    return storage_name