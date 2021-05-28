import json
import os
from typing import List, Iterable, Callable, Dict, Tuple, Set

from dataset_specific.msmarco.common import QueryID, load_query_group
from epath import job_man_dir
from misc_lib import DataIDManager, exist_or_mkdir, select_one_pos_neg_doc
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.msmarco_doc_gen.fast_gen.seg_resource import SegmentResourceLoader, SRPerQueryDoc
from tlm.data_gen.msmarco_doc_gen.fast_gen.sr_to_tfrecord import encode_sr
from tlm.data_gen.msmarco_doc_gen.missing_resource import missing_qids


class BestSegmentPredictionGen:
    def __init__(self,
                 max_seq_length,
                 split,
                 skip_single_seg,
                 pick_for_pairwise,
                 out_dir):
        self.query_group: List[List[QueryID]] = load_query_group(split)
        self.seg_resource_loader = SegmentResourceLoader(job_man_dir, split)
        self.max_seq_length = max_seq_length
        self.out_dir = out_dir
        self.skip_single_seg = skip_single_seg
        self.pick_for_pairwise = pick_for_pairwise
        self.info_dir = self.out_dir + "_info"
        exist_or_mkdir(self.info_dir)

    def work(self, job_id):
        qids = self.query_group[job_id]
        max_data_per_job = 1000 * 1000
        base = job_id * max_data_per_job
        data_id_manager = DataIDManager(base, base+max_data_per_job)
        output_path = os.path.join(self.out_dir, str(job_id))
        writer = RecordWriterWrap(output_path)

        for qid in qids:
            try:
                sr_per_qid = self.seg_resource_loader.load_for_qid(qid)
                docs_to_predict = select_one_pos_neg_doc(sr_per_qid.sr_per_query_doc)
                for sr_per_doc in docs_to_predict:
                    label_id = sr_per_doc.label
                    if self.skip_single_seg and len(sr_per_doc.segs) == 1:
                        continue
                    for seg_idx, seg in enumerate(sr_per_doc.segs):
                        info = {
                            'qid': qid,
                            'doc_id': sr_per_doc.doc_id,
                            'seg_idx': seg_idx
                        }
                        data_id = data_id_manager.assign(info)
                        feature = encode_sr(seg,
                                            self.max_seq_length,
                                            label_id,
                                            data_id)
                        writer.write_feature(feature)
            except FileNotFoundError:
                if qid in missing_qids:
                    pass
                else:
                    raise

        writer.close()
        info_save_path = os.path.join(self.info_dir, "{}.info".format(job_id))
        json.dump(data_id_manager.id_to_info, open(info_save_path, "w"))

