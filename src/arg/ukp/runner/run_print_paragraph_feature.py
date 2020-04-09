import os
import pickle
from typing import List

from arg.pf_common.base import ParagraphFeature
from arg.pf_common.print_paragraph_feature import print_paragraph_feature
from base_type import FileName
from cpath import output_path, pjoin
from data_generator.job_runner import sydney_working_dir


def print_features():
    job_dir = "ukp_paragraph_feature_2"
    job_id = 0
    file_path = os.path.join(sydney_working_dir, job_dir, str(job_id))
    features: List[ParagraphFeature] = pickle.load(open(os.path.join(file_path), "rb"))

    out_path = pjoin(output_path, FileName("ukp_paragraph_feature_2.html"))
    print_paragraph_feature(features, out_path)


if __name__ == "__main__":
    print_features()