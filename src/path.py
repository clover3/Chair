import os
from os.path import dirname
from misc_lib import exist_or_mkdir

project_root = os.path.abspath(dirname(dirname((os.path.abspath(__file__)))))
data_path = os.path.join(project_root, 'data')
exist_or_mkdir(data_path)
cache_path = os.path.join(data_path, 'cache')
exist_or_mkdir(cache_path)
output_path = os.path.join(project_root, 'output')
log_path = os.path.join(project_root, 'common.log')

model_path = os.path.join(output_path, 'model')


prediction_dir = os.path.join(output_path, "prediction")


def get_model_full_path(exp_name, run_id = None):
    run_dir = os.path.join(model_path, 'runs')
    save_dir = os.path.join(run_dir, exp_name)
    model_id = None
    if run_id is None:
        for (dirpath, dirnames, filenames) in os.walk(save_dir):
            for filename in filenames:
                if ".meta" in filename:
                    print(filename)
                    model_id = filename[:-5]
    else:
        model_id = "model-{}".format(run_id)

    if model_id is None:
        raise FileNotFoundError("Model does not exist on :", save_dir)
    return os.path.join(save_dir, model_id)


def get_latest_model_path(exp_name):
    run_dir = os.path.join(model_path, 'runs')
    save_dir = os.path.join(run_dir, exp_name)
    return get_latest_model_path_from_dir_path(save_dir)


def get_latest_model_path_from_dir_path(save_dir):
    max_id = ""
    max_step = 0
    for (dirpath, dirnames, filenames) in os.walk(save_dir):
        for filename in filenames:
            if ".meta" in filename:
                model_id = filename[:-5]
                step = int(model_id.split("-")[1])
                if step > max_step:
                    max_step = max_step
                    max_id = model_id

    if not max_id:
        return ""
    else:
        return os.path.join(save_dir, max_id)


def get_bert_full_path():
    return os.path.join(model_path, 'runs', "uncased_L-12_H-768_A-12", "bert_model.ckpt")



def open_pred_output(name):
    path = os.path.join(prediction_dir, name)
    fout = open(path, "w")
    return fout
