
from tlm.retrieve_lm.per_doc_posting_server import get_reader




reader = get_reader()

t = reader.retrieve("FBIS3-21572")
print(t)