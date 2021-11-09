from data_generator.job_runner import JobRunner
from epath import job_man_dir
from tlm.data_gen.adhoc_datagen import FirstSegmentAsDoc
from tlm.data_gen.msmarco_doc_gen.gen_worker import PredictionAllPassageGenerator
from tlm.data_gen.msmarco_doc_gen.mmd_worker import MMDWorker
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResourcePredict

if __name__ == "__main__":
    split = "dev"
    resource = ProcessedResourcePredict(split)
    max_seq_length = 512
    basic_encoder = FirstSegmentAsDoc(max_seq_length)
    generator = PredictionAllPassageGenerator(resource, basic_encoder, max_seq_length)

    def factory(out_dir):
        return MMDWorker(resource.query_group, generator, out_dir)

    runner = JobRunner(job_man_dir, len(resource.query_group)-1, "MMD_pred_first_{}".format(split), factory)
    runner.start()
