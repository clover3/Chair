from data_generator.job_runner import JobRunner
from epath import job_man_dir
from tlm.data_gen.adhoc_datagen import FirstAndRandom
from tlm.data_gen.msmarco_doc_gen.gen_worker import PointwiseGen
from tlm.data_gen.msmarco_doc_gen.mmd_worker import MMDWorker
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResource10doc

if __name__ == "__main__":
    split = "train"
    resource = ProcessedResource10doc(split)
    max_seq_length = 512
    basic_encoder = FirstAndRandom(max_seq_length, -1)
    generator = PointwiseGen(resource, basic_encoder, max_seq_length)

    def factory(out_dir):
        return MMDWorker(resource.query_group, generator, out_dir)

    runner = JobRunner(job_man_dir, len(resource.query_group)-1, "MMD_first_and_random".format(split), factory)
    runner.start()
