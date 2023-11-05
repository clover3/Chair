from taskman_client.wrapper3 import JobContext


def main():
    job_no = int(sys.argv[1])
    del_rate = 0.5
    dataset_name = "mmp_pep_es1"

    with JobContext(f"mmp_pep_es1_{job_no}"):
        pass


if __name__ == "__main__":
    main()
