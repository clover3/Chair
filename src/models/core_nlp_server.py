import os

from nltk.parse.corenlp import CoreNLPServer

# The server needs to know the location of the following files:
#   - stanford-corenlp-X.X.X.jar
#   - stanford-corenlp-X.X.X-models.jar
from cpath import data_path

# Create the server
server = CoreNLPServer(
   os.path.join(data_path, "stanford-corenlp-4.0.0.jar"),
   os.path.join(data_path, "stanford-corenlp-4.0.0-models.jar"),
)

# Start the server in the background
server.start()
