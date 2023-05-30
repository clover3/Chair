import time

from taskman_client.wrapper3 import JobContext


def main():
    # Measure difference of two given ranked list
    run_name = "job_context_dev"
    job_context = JobContext(run_name)
    with JobContext(run_name):
        print("Started job")
        time.sleep(2)

        raise ValueError()
        print("Successful termination")


if __name__ == "__main__":
    main()