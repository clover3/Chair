from arg.perspectives.random_walk.random_walk_worker import RandomWalkWorker
from data_generator.job_runner import JobRunner, sydney_working_dir

if __name__ == "__main__":
    def worker_gen(out_dir):
        return RandomWalkWorker(out_dir, "pc_dev_co_occur")

    runner = JobRunner(sydney_working_dir, 112, "random_walk", worker_gen)
    runner.start()

