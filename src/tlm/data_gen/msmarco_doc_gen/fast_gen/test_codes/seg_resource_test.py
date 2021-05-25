from epath import job_man_dir
from tlm.data_gen.msmarco_doc_gen.fast_gen.seg_resource import SegmentResourceLoader, SRPerQuery


def seg_resource_loader_test():
    srl = SegmentResourceLoader(job_man_dir, "train")
    qid = "1000008"
    sr_per_query: SRPerQuery = srl.load_for_qid(qid)

    assert qid == sr_per_query.qid
    for sr in sr_per_query.sr_per_query_doc:
        print(sr.doc_id)
        print(len(sr.segs))
        for s in sr.segs:
            print(s.first_seg)
            print(s.second_seg)
            break
        break


def main():
    seg_resource_loader_test()


if __name__ == "__main__":
    main()