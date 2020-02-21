from data_generator.argmining import ukp
from data_generator.job_runner import JobRunner, sydney_working_dir
from tlm.data_gen.run_sero_gen_wiki import SeroGen
from tlm.ukp.data_gen.run_ukp_gen1 import UkpWorker


if __name__ == "__main__":
    top_k = 1000
    num_jobs = len(ukp.all_topics) - 1
    target_seq_len = (128 - 2) * 24
    generator = SeroGen(target_seq_len, False)
    JobRunner(sydney_working_dir, num_jobs, "ukp_sero", lambda x: UkpWorker(x, top_k, generator)).start()


