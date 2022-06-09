import os

from cpath import output_path
from google_wrap.gs_wrap import upload_dir


def upload_nli_sg_files(data_name):
    local_dir_path = os.path.join(output_path, "tfrecord", data_name)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\work\Code\webtool\CloverTPU-3fa50b250c68.json"
    upload_dir(local_dir_path, "gs://clovertpu/training/data/" + data_name)