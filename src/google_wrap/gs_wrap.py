from google.cloud import storage
from subprocess import Popen, PIPE
import logging
import os
from misc_lib import exist_or_mkdir
from os.path import join as pjoin
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


def download_model_last(gs_model_dir, local_dir):
    print("Downloading model from :", gs_model_dir)
    client = storage.Client()
    latest = None

    gs_model_prefix = gs_model_dir + "/model"
    for blob in client.list_blobs("clovertpu", prefix=gs_model_prefix, ):
        print(blob.name)
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


