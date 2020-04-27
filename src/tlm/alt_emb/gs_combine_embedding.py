import glob
import os
import urllib.parse

from google.cloud import storage

from cpath import output_path
from google_wrap.gs_wrap import parse_gs_path
from misc_lib import exist_or_mkdir
from tlm.alt_emb.combine_embedding import combine

working_dir = os.path.join(output_path, "emb_combine_temp")
exist_or_mkdir(working_dir)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= "C:\work\Code\webtool\CloverTPU-3fa50b250c68.json"


def download(model_path):
    bucket_name, gs_path = parse_gs_path(model_path)
    base_name = os.path.basename(gs_path)
    dir_name = os.path.dirname(gs_path).replace("/", "_")
    local_dir_path = os.path.join(working_dir, dir_name)
    local_file_path = os.path.join(local_dir_path, base_name)
    if os.path.exists(local_dir_path):
        return local_file_path

    print("Downloading ", model_path)
    client = storage.Client()
    exist_or_mkdir(local_dir_path)
    for blob in client.list_blobs(bucket_name, prefix=gs_path):
        file_name = os.path.basename(urllib.parse.unquote(blob.path))
        print(file_name)

        save_path = os.path.join(local_dir_path, file_name)
        blob.download_to_filename(save_path)
    return local_file_path

# model_1 : classification task
# model_2 : alt emb checkpoint
def download_and_combine(model_1_path, model_2_path, save_path):
    model_1_local_path = download(model_1_path)
    model_2_local_path = download(model_2_path)
    combine(model_1_local_path, model_2_local_path, save_path)

    checkpoint_dir = os.path.dirname(save_path)
    checkpoint_name = os.path.basename(save_path)
    save_checkpoint_file(checkpoint_dir, checkpoint_name)


def upload_to_gs(local_save_path, gs_dir_path):
    bucket_name, gs_path_postfix = parse_gs_path(gs_dir_path)

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    print("Now uploading")
    for local_file_path in glob.glob(local_save_path + "*"):
        local_file_name = os.path.basename(local_file_path)
        gs_target_path = gs_path_postfix + "/" + local_file_name
        blob = bucket.blob(gs_target_path)
        blob.upload_from_filename(local_file_path)
        print("Uploaded at ", gs_target_path)

    gs_checkpoint_path = gs_path_postfix + "/" + "checkpoint"
    blob = bucket.blob(gs_checkpoint_path)
    local_dir_path = os.path.dirname(local_save_path)
    local_checkpoint_path = os.path.join(local_dir_path, "checkpoint")
    blob.upload_from_filename(local_checkpoint_path)


def save_checkpoint_file(dir_path, checkpoint_name):
    print(checkpoint_name)
    line = "model_checkpoint_path: \"{}\"\n".format(checkpoint_name)
    line += "all_model_checkpoint_paths: \"{}\"\n".format(checkpoint_name)
    path = os.path.join(dir_path, "checkpoint")
    open(path, "w").write(line)
