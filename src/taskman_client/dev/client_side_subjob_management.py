import time

from taskman_client.job_group_proxy import JobGroupProxy


def main():
    job_name = "dev_job_group"
    max_job = 10
    job_group = JobGroupProxy(job_name, max_job)

    with job_group.sub_job_context(3):
        time.sleep(10)
        raise Exception()



def manual_start():
    job_name = "dev_job_group"
    max_job = 10
    job_id = 0
    job_group = JobGroupProxy(job_name, max_job)
    # ret = job_group.sub_job_start(job_id)
    ret = job_group.sub_job_done(job_id)
    print(ret)



if __name__ == "__main__":
    main()