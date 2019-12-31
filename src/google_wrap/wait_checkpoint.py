import sys
import time

from google.cloud import storage


def wait_checkpoint(model_dir, step):
    found = check_checkpoint(model_dir, step)
    acc_sleep_time = 0
    sleep_interval = 60
    while not found:
        acc_sleep_time += sleep_interval
        print("\r Sleeping {} mins".format(int(acc_sleep_time/60)), end="")
        time.sleep(sleep_interval)
        found = check_checkpoint(model_dir, step)


def check_checkpoint(model_dir, step):
    model_dir_path = "training/model/" + model_dir
    target_path = model_dir_path + "/model.ckpt-{}".format(step)
    client = storage.Client()
    for blob in client.list_blobs("clovertpu", prefix=target_path):
        print(blob.name)
        return True
    return False


if __name__ == "__main__":
    wait_checkpoint(sys.argv[1], sys.argv[2])
