import os


def get_storage_name():
    if "storage_name" in os.environ:
        storage_name = os.environ["storage_name"]
    else:
        default_name = "clovertpu"
        print("Env var 'storage name' not found. default to ", default_name)
        storage_name = default_name
    return storage_name