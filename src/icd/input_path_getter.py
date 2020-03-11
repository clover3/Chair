import os

from cpath import data_path


def get_icm10cm_order_path():
    path = os.path.join(data_path, "2020_Code_Descriptions", "icd10cm_order_2020.txt")
    return path
