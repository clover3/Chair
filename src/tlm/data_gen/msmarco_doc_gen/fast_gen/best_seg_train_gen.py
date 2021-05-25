import os
from typing import List, Dict, Tuple

from dataset_specific.msmarco.common import QueryID, load_query_group
from epath import job_man_dir
from misc_lib import exist_or_mkdir, TimeProfiler
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.msmarco_doc_gen.fast_gen.collect_best_seg_prediction import BestSegCollector
from tlm.data_gen.msmarco_doc_gen.fast_gen.seg_resource import SegmentResourceLoader
from tlm.data_gen.msmarco_doc_gen.fast_gen.sr_to_tfrecord import encode_sr
from tlm.data_gen.msmarco_doc_gen.missing_resource import missing_qids


class BestSegTrainGen:
    def __init__(self, max_seq_length,
                 best_seg_collector: BestSegCollector,
                 out_dir):
        self.query_group: List[List[QueryID]] = load_query_group("train")
        self.seg_resource_loader = SegmentResourceLoader(job_man_dir, "train")
        self.max_seq_length = max_seq_length
        self.out_dir = out_dir
        self.info_dir = self.out_dir + "_info"
        self.best_seg_collector = best_seg_collector
        exist_or_mkdir(self.info_dir)

    def work(self, job_id):
        qid_to_max_seg_idx: Dict[Tuple[str, str], int] = self.best_seg_collector.get_best_seg_info(job_id)
        qids = self.query_group[job_id]
        output_path = os.path.join(self.out_dir, str(job_id))
        writer = RecordWriterWrap(output_path)
        for qid in qids:
            sr_per_qid = self.seg_resource_loader.load_for_qid(qid)
            for sr_per_doc in sr_per_qid.sr_per_query_doc:
                if len(sr_per_doc.segs) == 1:
                    continue
                qdid = qid, sr_per_doc.doc_id
                max_seg_idx = qid_to_max_seg_idx[qdid]
                label_id = sr_per_doc.label
                try:
                    seg = sr_per_doc.segs[max_seg_idx]
                    feature = encode_sr(seg,
                                        self.max_seq_length,
                                        label_id,
                                        )
                    writer.write_feature(feature)
                except IndexError:
                    print('qid={} doc_id={}'.format(qid, sr_per_doc.doc_id))
                    print("max_seg_idx={} but len(segs)={}".format(max_seg_idx, len(sr_per_doc.segs)))
                    raise

        writer.close()


class SingleSegTrainGen:
    def __init__(self, max_seq_length,
                 out_dir):
        self.query_group: List[List[QueryID]] = load_query_group("train")
        self.seg_resource_loader = SegmentResourceLoader(job_man_dir, "train")
        self.max_seq_length = max_seq_length
        self.out_dir = out_dir
        self.info_dir = self.out_dir + "_info"
        exist_or_mkdir(self.info_dir)

    def work(self, job_id):
        qids = self.query_group[job_id]
        output_path = os.path.join(self.out_dir, str(job_id))
        writer = RecordWriterWrap(output_path)
        for qid in qids:
            try:
                sr_per_qid = self.seg_resource_loader.load_for_qid(qid)
                for sr_per_doc in sr_per_qid.sr_per_query_doc:
                    if len(sr_per_doc.segs) > 1:
                        continue
                    label_id = sr_per_doc.label
                    seg = sr_per_doc.segs[0]
                    feature = encode_sr(seg,
                                        self.max_seq_length,
                                        label_id,
                                        )
                    writer.write_feature(feature)
            except FileNotFoundError:
                if qid in missing_qids:
                    pass
                else:
                    raise
        writer.close()
