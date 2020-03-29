import pickle
from typing import List

from arg.perspectives.select_paragraph import ParagraphClaimPersFeature
from base_type import FileName
from cpath import pjoin
from data_generator.job_runner import sydney_working_dir

if __name__ == "__main__":
    input_job_name: FileName = FileName("perspective_paragraph_feature_dev")
    input_dir = pjoin(sydney_working_dir, input_job_name)
    job_id = 0
    features: List[ParagraphClaimPersFeature] = pickle.load(open(pjoin(input_dir, FileName(str(job_id))), "rb"))
    print("Cid: ", features[0].claim_pers.cid)
