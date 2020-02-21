from crs.contradiction_pair.pair_payload import PairEncoder
from data_generator.job_runner import JobRunner, sydney_working_dir
from tlm.ukp.data_gen.clueweb12_13B import CluewebLMWorker

if __name__ == "__main__":
    generator = PairEncoder()
    JobRunner(sydney_working_dir, 1000, "clueweb12_13B_pair", lambda x: CluewebLMWorker(x, generator)).start()


