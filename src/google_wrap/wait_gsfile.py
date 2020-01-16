import sys

from google_wrap.wait_checkpoint import wait_gsfile

if __name__ == "__main__":
    wait_gsfile(sys.argv[1])
