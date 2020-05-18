from arg.perspectives.random_walk.random_walk_worker import RandomWalkWorker
from data_generator.job_runner import JobRunner, sydney_working_dir

if __name__ == "__main__":
    def worker_gen(out_dir):
        return RandomWalkWorker(out_dir, "dev_claim_graph")

    runner = JobRunner(sydney_working_dir, 139, "dev_claim_random_walk_debug2", worker_gen)
    runner.start()

