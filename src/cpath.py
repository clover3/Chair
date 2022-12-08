import os
from os.path import dirname

from base_type import FilePath
from misc_lib import exist_or_mkdir

project_root = FilePath(os.path.abspath(dirname(dirname((os.path.abspath(__file__))))))
data_path = FilePath(os.path.join(project_root, 'data'))
src_path = FilePath(os.path.join(project_root, 'src'))
exist_or_mkdir(data_path)
cache_path = FilePath(os.path.join(data_path, 'cache'))
exist_or_mkdir(cache_path)
json_cache_path = FilePath(os.path.join(data_path, 'json'))
exist_or_mkdir(json_cache_path)
output_path = FilePath(os.path.join(project_root, 'output'))
log_path = FilePath(os.path.join(project_root, 'common.log'))

common_model_dir_root = FilePath(os.path.join(output_path, 'model'))
prediction_dir = FilePath(os.path.join(output_path, "prediction"))

qtype_root_dir = FilePath(os.path.join(output_path, "qtype"))
word2vec_path = os.path.join(data_path, "GoogleNews-vectors-negative300.bin")


def pjoin(file_path: FilePath, file_name) -> FilePath:
    return FilePath(os.path.join(file_path, file_name))


def get_model_full_path(exp_name, run_id = None):
    run_dir = os.path.join(common_model_dir_root, 'runs')
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
    run_dir = os.path.join(common_model_dir_root, 'runs')
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
                    max_step = step
                    max_id = model_id

    if not max_id:
        return False
    else:
        return os.path.join(save_dir, max_id)


def get_bert_full_path():
    return os.path.join(common_model_dir_root, 'runs', "uncased_L-12_H-768_A-12", "bert_model.ckpt")


def get_bert_config_path():
    return os.path.join(data_path, 'bert_config.json')


def open_pred_output(name):
    path = os.path.join(prediction_dir, name)
    fout = open(path, "w")
    return fout


def at_output_dir(folder_name, file_name):
    return os.path.join(output_path, folder_name, file_name)


def at_data_dir(folder_name, file_name):
    return os.path.join(data_path, folder_name, file_name)


def get_canonical_model_path(name):
    run_dir = os.path.join(common_model_dir_root, 'runs')
    save_dir = os.path.join(run_dir, name)

    exist_or_mkdir(common_model_dir_root)
    exist_or_mkdir(run_dir)
    exist_or_mkdir(save_dir)
    return save_dir


def get_canonical_model_path2(name, sub_dir):
    run_dir = os.path.join(common_model_dir_root, 'runs')
    save_dir = os.path.join(run_dir, name, sub_dir)

    exist_or_mkdir(common_model_dir_root)
    exist_or_mkdir(run_dir)
    exist_or_mkdir(save_dir)
    return save_dir


