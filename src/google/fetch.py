from google import gsutil




def fetch_bert():
    model_step = 20000
    dir_path = "gs://clovertpu/training/gun_control"
    save_name = "gun_control_20000"
    load_id = gsutil.download_model(dir_path, model_step, save_name)
    return load_id


if __name__ == '__main__':
    fetch_bert()