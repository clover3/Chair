
import os
import pickle
import re
from os.path import dirname

project_root = os.path.abspath(dirname(dirname(dirname(os.path.abspath(__file__)))))
data_path = os.path.join(project_root, 'data')
