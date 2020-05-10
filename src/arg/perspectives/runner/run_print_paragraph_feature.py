import os
import pickle
from typing import List

from arg.perspectives.PerspectiveParagraphTFRecordWorker import to_paragraph_feature
from arg.perspectives.declaration import ParagraphClaimPersFeature
from arg.pf_common.base import ParagraphFeature
from arg.pf_common.print_paragraph_feature import print_paragraph_feature
from base_type import FileName
from cpath import output_path, pjoin
from data_generator.job_runner import sydney_working_dir
from list_lib import lmap


def print_features():
    job_dir = "perspective_paragraph_feature"
    job_id = 0
    file_path = os.path.join(sydney_working_dir, job_dir, str(job_id))

    features: List[ParagraphClaimPersFeature] = pickle.load(open(os.path.join(file_path), "rb"))
    features: List[ParagraphFeature] = lmap(to_paragraph_feature, features)

    out_path = pjoin(output_path, FileName("perspective_paragraph_feature.html"))
    print_paragraph_feature(features, out_path)


if __name__ == "__main__":
    print_features()
