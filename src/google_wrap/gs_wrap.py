import os
from os.path import join as pjoin

from google.cloud import storage

from misc_lib import exist_or_mkdir
from path import model_path


def credential_check():
    if not os.environ["GOOGLE_APPLICATION_CREDENTIALS"]:
        raise Exception("GOOGLE_APPLICATION_CREDENTIALS not found")


GS_MODEL_ROOT_DIR = "training/model"

def get_step_from_path(gs_path):
    tokens = gs_path.split("/")
    t = tokens[-1].split(".")[1]
    step_str = t.split("-")[1]
    return step_str

def drop_after_step_str(file_path):
    for key in [".data", ".meta", ".index"]:
        idx = file_path.find(key)
        if idx > 0:
            return file_path[:idx]

    raise Exception("Failed to parse the file name")

def get_last_model_path(model_dir_path):
    client = storage.Client()
    latest = None

    for blob in client.list_blobs("clovertpu", prefix=model_dir_path):
        if latest is None or latest.updated < blob.updated:
            latest = blob

    if latest is None:
        raise FileNotFoundError(model_dir_path)

    return drop_after_step_str(latest.name)


def download_model_last(gs_model_dir, local_dir):
    print("Downloading model from :", gs_model_dir)
    client = storage.Client()
    latest = None

    gs_model_prefix = gs_model_dir + "/model"
    for blob in client.list_blobs("clovertpu", prefix=gs_model_prefix, ):
        if latest is None or latest.updated < blob.updated:
            latest = blob

    if latest is None:
        raise FileNotFoundError(gs_model_dir)

    step = get_step_from_path(latest.name)
    download_prefix = gs_model_prefix + ".ckpt-{}".format(step)
    for blob in client.list_blobs("clovertpu", prefix=download_prefix):
        postfix = blob.name.split("/")[-1]
        local_path = pjoin(local_dir, postfix)
        r = blob.download_to_filename(local_path)

    checkpoint_path = gs_model_dir + "/checkpoint"
    for blob in client.list_blobs("clovertpu", prefix=checkpoint_path):
        local_path = pjoin(local_dir, "checkpoint")
        blob.download_to_filename(local_path)

    print("Download completed")
    return step

# returns <str:step>
def download_model_last_auto(run_name):
    gs_path = GS_MODEL_ROOT_DIR + "/" + run_name
    local_path = pjoin(model_path, "runs", run_name)
    exist_or_mkdir(local_path)
    steps = download_model_last(gs_path, local_path)
    return run_name, "model.ckpt-{}".format(steps)


def is_full_checkpoint_path(init_checkpoint):
    if init_checkpoint.endswith("bert_model.ckpt"):
        return True
    try:
        last_token = init_checkpoint.split("-")[-1]
        if int(last_token) > 0:
            pass
    except Exception as e:
        return False
    return True


def auto_resolve_init_checkpoint(init_checkpoint):
    if init_checkpoint is None:
        return init_checkpoint
    elif is_full_checkpoint_path(init_checkpoint):
        return init_checkpoint
    else:
        assert "/" not in init_checkpoint
        print("Resolving init_checkpoint : ", init_checkpoint)
        def get_last_model_path_from_dir_name(model_dir):
            model_dir_path = "training/model/" + model_dir
            return get_last_model_path(model_dir_path)

        return get_last_model_path_from_dir_name(init_checkpoint)

