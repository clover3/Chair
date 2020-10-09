import sys
import time

from google.cloud import storage

# model.ckpt-1000.meta
from google_wrap.get_storage_name import get_storage_name


def get_name_from_path(file_path):
    tokens = file_path.split("/")
    name = tokens[-1]
    return name

def check_new_checkpoint(known_checkpoint, new_files):
    new_checkpoint_set = set()
    for file_path in new_files:
        name = get_name_from_path(file_path)

        try:
            if name.startswith("model.ckpt-"):
                model, ckpt_step, postfix = name.split(".")
                print(model, ckpt_step, postfix)
                step = int(ckpt_step[len("ckpt-"):])

                if postfix == "meta" or postfix == "index":
                    checkpoint_name = ".".join([model, ckpt_step])
                    if checkpoint_name not in known_checkpoint:
                        new_checkpoint_set.add(checkpoint_name)

        except Exception as e:
            print(name, "is not valid")
            pass
    return new_checkpoint_set


def is_valid_checkpoint(checkpoint_name, info_list):
    all_names = list([get_name_from_path(info['name']) for info in info_list])
    if not (checkpoint_name + ".meta" in all_names and checkpoint_name + ".index" in all_names):
        return False

    for name in all_names:
        if name.startswith(checkpoint_name + ".data"):
            return True
        else:
            return False


def watch_dir(model_dir):
    sleep_interval = 1
    stop = False
    last_info = {}
    known_checkpoint = None
    while not stop:
        info_list = get_file_list(model_dir)

        if last_info is not None:
            new_files = find_new_files(info_list, last_info)
            new_checkpoint_names = check_new_checkpoint(known_checkpoint, new_files)
            valid_checkpoints = list([name for name in new_checkpoint_names if is_valid_checkpoint(name, info_list)])
            if valid_checkpoints:
                print("New valid checkpoints: ", valid_checkpoints)
                known_checkpoint.extend(valid_checkpoints)
                stop = True
        last_info = {}
        for info in info_list:
            last_info[info['name']] = info

        time.sleep(sleep_interval)


def find_new_files(info_list, last_info):
    new_files = []
    for info in info_list:
        if info['name'] in last_info:
            old_info = last_info[info['name']]
            for key in info:
                if info[key] != old_info[key]:
                    print("{}'s {} changed from {} to {}".format(info['name'], key, info[key], old_info[key]))
        else:
            new_files.append(info['name'])
            print("new file :", info['name'])
    return new_files


def get_model_path(model_dir, step):
    model_dir_path = "training/model/" + model_dir
    target_path = model_dir_path + "/model.ckpt-{}".format(step)
    return target_path


def get_file_list(target_path):
    if "clovertpu" in target_path:
        print("WARNING, target path should not include the root storage name")
    client = storage.Client()

    def fetch_info(blob):
        d = {
            'name': blob.name,
            'size': blob.size,
            'updated_time':blob.updated,
            'crc': blob.crc32c,
            'id':blob.id
        }
        return d
    info_list = list(map(fetch_info, client.list_blobs(get_storage_name(), prefix=target_path)))
    return info_list


if __name__ == "__main__":
    watch_dir(sys.argv[1])
