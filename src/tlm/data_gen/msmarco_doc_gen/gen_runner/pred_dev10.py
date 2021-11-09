from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS
from tlm.data_gen.adhoc_datagen import AllSegmentAsDoc
from tlm.data_gen.msmarco_doc_gen.gen_worker import PredictionAllPassageGenerator
from tlm.data_gen.msmarco_doc_gen.mmd_worker import MMDWorker
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResourcePredict10

if __name__ == "__main__":
    split = "dev"
    resource = ProcessedResourcePredict10(split)
    max_seq_length = 512
    basic_encoder = AllSegmentAsDoc(max_seq_length)
    generator = PredictionAllPassageGenerator(resource, basic_encoder, max_seq_length)

    def factory(out_dir):
        return MMDWorker(resource.query_group, generator, out_dir)

    runner = JobRunnerS(job_man_dir, len(resource.query_group)-1, "MMD_pred10".format(split), factory)
    runner.start()
