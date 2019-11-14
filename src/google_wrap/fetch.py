from google import gsutil

import time


def fetch_bert(topic):
    model_step = 20000
    dir_path = "gs://clovertpu/training/{}".format(topic)
    save_name = "{}_20000".format(topic)
    load_id = gsutil.download_model(dir_path, model_step, save_name)
    return load_id


if __name__ == '__main__':
    time.sleep(3600 * 8)
    fetch_bert("school_uniforms")
    #fetch_bert("marijuana_legalization")