import data_generator.argmining.ukp_header
from data_generator.job_runner import JobRunner, sydney_working_dir
from tlm.data_gen.run_sero_gen_wiki import SeroGen
from tlm.ukp.data_gen.run_ukp_gen2 import UkpWorker2


if __name__ == "__main__":
    num_jobs = len(data_generator.argmining.ukp_header.all_topics) - 1
    target_seq_len = (128 - 2) * 24
    top_k = 10000
    generator = SeroGen(target_seq_len, False)
    JobRunner(sydney_working_dir, num_jobs, "ukp_sero_10K", lambda x: UkpWorker2(x, generator, top_k)).start()


