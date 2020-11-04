import sys
import time

from google.cloud import storage

from google_wrap.get_storage_name import get_storage_name
from google_wrap.monitor_checkpoint import get_file_list, is_valid_checkpoint


def wait_checkpoint(model_dir, step):
    target_path = get_model_path(model_dir, step)
    print("Target path : ", target_path)
    found = check_gsfile_exists(target_path)
    acc_sleep_time = 0
    sleep_interval = 60
    while not found:
        acc_sleep_time += sleep_interval
        print("\r Sleeping {} mins".format(int(acc_sleep_time / 60)), end="")
        time.sleep(sleep_interval)
        if check_gsfile_exists(target_path):
            info_list = get_file_list(model_dir)
            found = is_valid_checkpoint(target_path, info_list)


def get_model_path(model_dir, step):
    model_dir_path = "training/model/" + model_dir
    target_path = model_dir_path + "/model.ckpt-{}".format(step)
    return target_path


def check_gsfile_exists(target_path):
    if "clovertpu" in target_path:
        print("WARNING, target path should not include the root storage name")
    client = storage.Client()
    for blob in client.list_blobs(get_storage_name(), prefix=target_path):
        print(blob.name)
        return True
    return False


if __name__ == "__main__":
    wait_checkpoint(sys.argv[1], sys.argv[2])
