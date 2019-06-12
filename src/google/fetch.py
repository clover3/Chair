from google import gsutil




def fetch_bert():
    model_step = 80000
    dir_path = "gs://clovertpu/training/model_B_1e4"
    save_name = "Abortion_B_1e4_80000"
    load_id = gsutil.download_model(dir_path, model_step, save_name)
    return load_id


if __name__ == '__main__':
    fetch_bert()