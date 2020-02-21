import logging
import logging
import sys


def example():
    # create logger with 'spam_application'
    logger = logging.getLogger('spam_application')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Spam application log")

def fun(args):
    example()
    #tf_logging.info("This is TF log")
    #set_level_debug()
    #tf_logging.info("This is TF DEBUG")
    print("this is print ")


parser = argparse.ArgumentParser(description='File should be stored in ')
parser.add_argument("--start_model_path", help="Your input file.")
parser.add_argument("--start_type")
parser.add_argument("--save_dir")
parser.add_argument("--modeling_option")
parser.add_argument("--num_gpu", default=1)


if __name__  == "__main__":
    args = parser.parse_args(sys.argv[1:])
    fun(args)
