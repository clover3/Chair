import time

from job_manager.job_runner_with_server import JobRunnerWServer


class DummyWorker:
    def __init__(self, out_path):
        pass

    def work(self, job_id):
        time.sleep(1)

def main():
    working_path = "d:\\job_dir"
    max_job = 3
    job_name = "test_job"
    job_runner = JobRunnerWServer(working_path, max_job, job_name, DummyWorker)
    job_id = job_runner.pool_job()
    print(job_id)



if __name__ == "__main__":
    main()