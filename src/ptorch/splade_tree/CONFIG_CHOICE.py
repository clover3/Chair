import os

CONFIG_FULLPATH = os.environ["SPLADE_CONFIG_PATH"]
# CONFIG_PATH = os.environ["SPLADE_CONFIG_PATH"]
# CONFIG_NAME = os.environ["SPLADE_CONFIG_NAME"]
CONFIG_PATH, CONFIG_NAME = os.path.split(CONFIG_FULLPATH)

if ".yaml" in CONFIG_NAME:
    CONFIG_NAME = CONFIG_NAME.split(".yaml")[0]
