import os
import pickle
from collections import OrderedDict
from typing import List

from arg.perspectives.declaration import ParagraphClaimPersFeature
from arg.pf_common.base import ParagraphFeature, TPDataPoint, DPID
from arg.pf_common.encode_paragraph_feature_to_tfrecord import format_paragraph_features
from data_generator.job_runner import sydney_working_dir
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import foreach
from misc_lib import exist_or_mkdir
from tf_util.record_writer_wrap import RecordWriterWrap


def to_paragraph_feature(f: ParagraphClaimPersFeature) -> ParagraphFeature:
    return ParagraphFeature(feature=f.feature,
                          datapoint=TPDataPoint(
                              text1=f.claim_pers.claim_text,
                              text2=f.claim_pers.p_text,
                              id=DPID("{}_{}".format(f.claim_pers.cid, f.claim_pers.pid)),
                              label=int(f.claim_pers.label)
                            )
                         )


class PerspectiveParagraphTFRecordWorker:
    def __init__(self, input_job_name, out_dir):
        self.out_dir = out_dir
        exist_or_mkdir(out_dir)
        self.tokenizer = get_tokenizer()
        self.max_seq_length = 512
        self.input_dir = os.path.join(sydney_working_dir, input_job_name)

    def work(self, job_id):
        features: List[ParagraphClaimPersFeature] = pickle.load(open(os.path.join(self.input_dir, str(job_id)), "rb"))

        writer = RecordWriterWrap(os.path.join(self.out_dir, str(job_id)))
        for f in features:
            f2: ParagraphFeature = to_paragraph_feature(f)
            encoded_list: List[OrderedDict] = format_paragraph_features(self.tokenizer, self.max_seq_length, f2)
            foreach(writer.write_feature, encoded_list)
        writer.close()