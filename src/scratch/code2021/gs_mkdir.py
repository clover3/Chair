import tensorflow as tf


def main():
    output_dir = "gs://clovertpu/training/model/gs_debug2"
    tf.io.gfile.makedirs(output_dir)


if __name__ == "__main__":
    main()
