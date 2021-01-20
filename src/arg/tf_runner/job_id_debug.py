from my_tf import tf
from taskman_client.wrapper import report_run


@report_run
def main(_):
    print("Start main()")


if __name__ == "__main__":
    tf.compat.v1.app.run()
