from taskman_client.wrapper2 import report_run_named
import time


@report_run_named("test_run")
def main():
    print("Sleeping")
    time.sleep(10)
    print("Done")
    return NotImplemented


if __name__ == "__main__":
    main()