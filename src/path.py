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


def open_pred_output(name):
    path = os.path.join(prediction_dir, name)
    fout = open(path, "w")
    return fout
