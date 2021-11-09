from misc_lib import tprint, DataIDManager
from tlm.data_gen.adhoc_datagen import LeadingN
from tlm.data_gen.msmarco_doc_gen.gen_worker import PassageLengthInspector
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResource


def main():
    split = "train"
    resource = ProcessedResource(split)
    data_id_manager = DataIDManager(0)
    max_seq_length = 512
    basic_encoder = LeadingN(max_seq_length, 1)
    generator = PassageLengthInspector(resource, basic_encoder, max_seq_length)

    qids_all = []
    for job_id in range(40):
        qids = resource.query_group[job_id]
        data_bin = 100000
        data_id_st = job_id * data_bin
        data_id_ed = data_id_st + data_bin
        qids_all.extend(qids)

    tprint("generating instances")
    insts = generator.generate(data_id_manager, qids_all)
    generator.write(insts, "")


if __name__ == "__main__":
    main()
