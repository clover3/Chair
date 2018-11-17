import os
from os.path import dirname

project_root = os.path.abspath(dirname(dirname((os.path.abspath(__file__)))))
data_path = os.path.join(project_root, 'data')
output_path = os.path.join(project_root, 'output')
log_path = os.path.join(project_root, 'common.log')

model_path = os.path.join(output_path, 'model')