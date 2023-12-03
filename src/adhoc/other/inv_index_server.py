import sys
from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
import socketserver
from typing import List, Iterable, Callable, Dict, Tuple, Set

from cache import load_pickle_from
from trainer_v2.chair_logging import c_log

PORT_INV_INDEX = 8132


class InvIndexServer:
    def __init__(self, inv_index: Dict[str, List]):
        self.inv_index = inv_index

    def start(self, port=PORT_INV_INDEX):
        class RequestHandler(SimpleXMLRPCRequestHandler):
            rpc_paths = ('/RPC2',)

        class RPCThreading(socketserver.ThreadingMixIn, SimpleXMLRPCServer):
            pass

        c_log.info("Preparing server")
        server = RPCThreading(("0.0.0.0", port),
                              requestHandler=RequestHandler,
                              allow_none=True,
                              )
        server.register_introspection_functions()

        server.register_function(self.get_postings, 'get_postings')
        c_log.info("Waiting")
        server.serve_forever()

    def get_postings(self, key) -> List:
        try:
            ret = self.inv_index[key]
        except KeyError:
            ret = []
        c_log.info("Return {} items".format(len(ret)))
        return ret


def main():
    c_log.info("Loading pickles")
    inv_index = load_pickle_from(sys.argv[1])
    server = InvIndexServer(inv_index)
    server.start()


if __name__ == "__main__":
    main()